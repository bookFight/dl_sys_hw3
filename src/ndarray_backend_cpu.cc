#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

/**
 * 用于在内存中维护对齐到 ALIGNMENT 边界的数组的工具结构。
 * 这种对齐要求至少为 TILE * ELEM_SIZE（每个元素大小与 tile 大小的乘积），
 * 不过默认情况下我们会将其设置得更大以满足更高的性能需求。
 */
struct AlignedArray {
    /**
     * 构造函数：创建指定大小的内存对齐数组
     * @param size 数组中元素的数量（不是字节数）
     *
     * 实现说明：
     * 使用 posix_memalign 分配内存，确保：
     * 1. 内存地址是 ALIGNMENT 的整数倍（满足对齐要求）
     * 2. 分配的总字节数为 size * ELEM_SIZE（元素数量 × 单个元素字节数）
     * 如果分配失败（ret != 0），则抛出 bad_alloc 异常
     */
    AlignedArray(const size_t size) {
        // 分配对齐的内存：(void**)&ptr 接收分配的地址，ALIGNMENT 是对齐边界，size*ELEM_SIZE 是总字节数
        int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
        if (ret != 0) throw std::bad_alloc();  // 内存分配失败时抛出异常
        this->size = size;  // 记录元素数量
    }

    /**
     * 析构函数：释放对齐的内存
     * 使用 free() 释放由 posix_memalign 分配的内存（必须配对使用）
     */
    ~AlignedArray() {
        free(ptr);  // 释放内存，避免泄漏
    }

    /**
     * 将指针转换为整数形式返回
     * @return 指针 ptr 的内存地址（以 size_t 类型表示）
     * 用途：通常用于调试或验证内存地址是否满足对齐要求
     */
    size_t ptr_as_int() {
        return (size_t)ptr;
    }

    scalar_t* ptr;  // 指向对齐内存的指针，存储 scalar_t 类型的元素
    size_t size;    // 数组中元素的数量（不是字节数）
};


/**
 * 递增多维数组索引，类似数字进位操作，用于遍历数组时生成下一个索引
 *
 * @param[in,out] index  当前索引（多维），函数会直接修改该向量
 * @param[in]     shape  数组各维度的大小，用于判断索引是否越界
 * @return        bool   索引递增后是否仍有效（未超出数组范围）
 *
 * 示例：
 *  shape = [2,3]（2行3列的二维数组）
 *  初始index = [0,0] → 递增后 [0,1] → 返回true
 *  当index = [0,2] → 递增后 [1,0] → 返回true
 *  当index = [1,2] → 递增后超出范围 → 返回false
 */
bool next_index(std::vector<int32_t>& index, const std::vector<int32_t>& shape) {
  // 处理空索引（空数组）的特殊情况
  if(index.size() == 0){
    return false;
  }

  // 从最后一个维度开始递增（类似最低位加1）
  index[index.size()-1]++;

  // 处理进位：从最后一维向前检查是否越界
  for(int i = index.size()-1; i >= 0; i--){
    // 当前维度索引超出范围，需要进位
    if(index[i] >= shape[i]){
      // 重置当前维度索引为0（类似进位后低位归0）
      index[i] = 0;

      if(i > 0){
        // 向前一个维度进位（高位加1）
        index[i-1]++;
      } else {
        // 已经是第一维，进位后超出数组范围，返回无效
        return false;
      }
    } else {
      // 当前维度未越界，无需继续进位，返回有效
      return true;
    }
  }

  // 理论上不会执行到这里，作为安全返回
  return false;
}

/**
 * 将多维索引转换为内存中的偏移量，用于定位元素实际存储位置
 *
 * @param[in] index   多维索引（如[1,2]表示第二行第三列）
 * @param[in] strides 各维度的步长（元素在该维度上相邻两个元素的内存间隔）
 * @param[in] offset  数组的基础偏移量（相对于内存起始地址的偏移）
 * @return    size_t  计算得到的总偏移量（可直接用于访问内存）
 *
 * 计算逻辑：
 *  总偏移量 = 基础偏移量 + Σ(索引[i] * 步长[i])
 *  示例：
 *  对于shape=[2,3]、strides=[3,1]的数组（行优先存储）
 *  index=[1,2] → 偏移量 = 0 + 1*3 + 2*1 = 5（即第6个元素，0基索引）
 */
size_t index_to_offset(const std::vector<int32_t>& index, const std::vector<int32_t>& strides, const size_t offset) {
  // 初始偏移量为数组的基础偏移量
  size_t res = offset;

  // 累加每个维度的索引×步长，得到总偏移
  for(int i = 0; i < index.size(); i++){
    res += index[i] * strides[i];
  }

  return res;
}


void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



/**
 * 将非连续存储的数组压缩为连续存储的数组
 *
 * @param[in]  a        输入数组（可能非连续存储）
 * @param[out] out      输出数组（连续存储的结果）
 * @param[in]  shape    输入数组的多维形状（各维度大小）
 * @param[in]  strides  输入数组的步长（各维度相邻元素的内存间隔）
 * @param[in]  offset   输入数组的基础偏移量（起始位置）
 *
 * 功能说明：
 *  按多维顺序遍历输入数组的所有元素，将其依次存储到输出数组中，
 *  使输出数组成为紧凑的连续内存块，忽略原数组的非连续布局。
 */
void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape, std::vector<int32_t> strides, size_t offset) {
  /// BEGIN SOLUTION
  // 初始化多维索引，从第一个元素（全0索引）开始遍历
  auto a_index = std::vector<int32_t>(shape.size(), 0);

  // 遍历输出数组的每个位置（需要填充的总元素数 = out->size）
  for (int out_index = 0; out_index < out->size; out_index++) {
    // 计算当前多维索引在输入数组中的内存偏移量
    size_t a_offset = index_to_offset(a_index, strides, offset);

    // 将输入数组中对应位置的元素复制到输出数组的连续位置
    out->ptr[out_index] = a.ptr[a_offset];

    // 生成下一个多维索引（类似进位操作，确保按顺序遍历所有元素）
    next_index(a_index, shape);
  }
  /// END SOLUTION
}


/**
 * 按元素级方式将输入数组的值设置到输出数组的指定位置
 *
 * @param[in]  a        输入数组（连续存储的源数据）
 * @param[out] out      输出数组（目标数组，可能非连续存储）
 * @param[in]  shape    输出数组的多维形状（确定索引范围）
 * @param[in]  strides  输出数组的步长（计算目标位置的内存偏移）
 * @param[in]  offset   输出数组的基础偏移量（起始位置）
 *
 * 功能说明：
 *  遍历输入数组a的每个元素，按多维索引顺序将其依次写入输出数组out的指定位置，
 *  支持非连续的目标位置（通过strides和offset控制），实现类似a[i] → out[indices[i]]的映射。
 */
void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape, std::vector<int32_t> strides, size_t offset) {
  /// BEGIN SOLUTION
  // 初始化输出数组的多维索引，从第一个位置（全0索引）开始
  auto out_index = std::vector<int32_t>(shape.size(), 0);

  // 遍历输入数组的每个元素（总数量 = a.size）
  for (int a_index = 0; a_index < a.size; a_index++) {
    // 计算当前多维索引在输出数组中的内存偏移量
    size_t out_offset = index_to_offset(out_index, strides, offset);

    // 将输入数组的当前元素写入输出数组的指定偏移位置
    out->ptr[out_offset] = a.ptr[a_index];

    // 更新输出数组的多维索引，准备下一个元素的写入位置
    next_index(out_index, shape);
  }
  /// END SOLUTION
}

/**
 * 将标量值设置到输出数组的指定范围内的所有位置
 *
 * @param[in]  size     需要设置的元素数量
 * @param[in]  val      要设置的标量值
 * @param[out] out      输出数组（目标数组，可能非连续存储）
 * @param[in]  shape    输出数组的多维形状（确定索引范围）
 * @param[in]  strides  输出数组的步长（计算目标位置的内存偏移）
 * @param[in]  offset   输出数组的基础偏移量（起始位置）
 *
 * 功能说明：
 *  按多维索引顺序遍历输出数组的指定范围（共size个元素），
 *  将所有位置的值设置为同一个标量val，支持非连续的目标位置。
 */
void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape, std::vector<int32_t> strides, size_t offset) {
  /// BEGIN SOLUTION
  // 初始化输出数组的多维索引，从第一个位置（全0索引）开始
  auto out_index = std::vector<int32_t>(shape.size(), 0);

  // 遍历需要设置的所有位置（总数量 = size）
  for (int i = 0; i < size; i++) {
    // 计算当前多维索引在输出数组中的内存偏移量
    size_t out_offset = index_to_offset(out_index, strides, offset);

    // 将标量值写入输出数组的指定偏移位置
    out->ptr[out_offset] = val;

    // 更新输出数组的多维索引，准备下一个位置的设置
    next_index(out_index, shape);
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

void EwiseOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out,
               std::function<scalar_t(scalar_t, scalar_t)> op) {
  /**
   * Apply an element-wise operation to two arrays and write the result to out.
   *
   * Args:
   *   a: compact array of size a.size
   *   b: compact array of size b.size
   *   out: compact array of size out.size to write the output to
   *   op: function pointer to the operation to apply
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}

void ScalarOp(const AlignedArray& a, scalar_t val, AlignedArray* out,
               std::function<scalar_t(scalar_t, scalar_t)> op) {
  /**
   * Apply an element-wise operation to an array and a scalar, and write the result to out.
   *
   * Args:
   *   a: compact array of size a.size
   *   val: scalar value to apply the operation with
   *   out: compact array of size out.size to write the output to
   *   op: function pointer to the operation to apply
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], val);
  }
}

/**
     * 将两个（紧凑存储的）矩阵相乘，结果写入输出（同样紧凑存储的）矩阵。
     * 本实现使用"朴素"的三重循环算法。
     *
     * 参数说明：
     *   a: 紧凑存储的二维数组，尺寸为 m x n（m行n列）
     *   b: 紧凑存储的二维数组，尺寸为 n x p（n行p列）
     *   out: 用于存储结果的紧凑二维数组，尺寸为 m x p（m行p列）
     *   m: 矩阵a和输出矩阵out的行数
     *   n: 矩阵a的列数 / 矩阵b的行数（矩阵相乘的维度匹配条件）
     *   p: 矩阵b和输出矩阵out的列数
     *
     * 注意：
     *   紧凑存储（compact）指矩阵元素按行优先顺序连续存储在内存中，
     *   即对于矩阵中的元素(i,j)，其在数组中的索引为 i*列数 + j
     */
void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /// BEGIN SOLUTION
  // 初始化输出矩阵所有元素为0（避免残留垃圾值影响累加结果）
  for(uint32_t i = 0; i < m*p; i++){
    out->ptr[i] = 0;
  }

  // 三重循环实现矩阵乘法：out[i][j] = sum_{k=0 to n-1} a[i][k] * b[k][j]
  // i：输出矩阵行索引（0到m-1）
  for (uint32_t i = 0; i < m; i++) {
    // j：输出矩阵列索引（0到p-1）
    for (uint32_t j = 0; j < p; j++) {
      // k：中间维度累加索引（0到n-1）
      for (uint32_t k = 0; k < n; k++) {
        // 计算元素在紧凑数组中的索引：
        // a[i][k] = a.ptr[i*n + k]（a的列数为n）
        // b[k][j] = b.ptr[k*p + j]（b的列数为p）
        // out[i][j] = out.ptr[i*p + j]（out的列数为p）
        out->ptr[i*p + j] += a.ptr[i*n + k] * b.ptr[k*p + j];
      }
    }
  }
  /// END SOLUTION
}


/**
     * 将两个TILE×TILE的矩阵相乘，并将结果**累加**到out中（重要：是累加而非覆盖，
     * 因此不应在之前将out设为零）。此处包含的编译器标志可使编译器正确使用向量操作来实现此函数。
     * 具体而言，__restrict__关键字向编译器表明a、b和out没有重叠的内存区域（这是向量操作与
     * 非向量化操作结果一致的必要条件——想象一下如果a、b和out内存重叠可能发生的问题）。
     * 类似地，__builtin_assume_aligned关键字告诉编译器输入数组将对齐到内存中的适当块，
     * 这也有助于编译器进行代码向量化。
     *
     * 参数说明：
     *   a: 紧凑存储的二维数组，尺寸为TILE×TILE
     *   b: 紧凑存储的二维数组，尺寸为TILE×TILE
     *   out: 紧凑存储的二维数组，尺寸为TILE×TILE，用于存储结果（累加操作）
     */
inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  // 告诉编译器数组a、b、out按TILE×ELEM_SIZE字节对齐，帮助优化向量操作
  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
    // 三重循环实现TILE×TILE矩阵乘法，并将结果累加到out中
    // i：结果矩阵行索引（0到TILE-1）
// 优化后的循环顺序：i→k→j
    for (uint32_t i = 0; i < TILE; ++i) {
        for (uint32_t k = 0; k < TILE; ++k) {  // 先循环k，再循环j
            float a_ik = a[i * TILE + k];  // 缓存a[i][k]，避免重复计算索引
            for (uint32_t j = 0; j < TILE; ++j) {
            out[i * TILE + j] += a_ik * b[k * TILE + j];  // b的访问变为连续行访问
        }
    }
}
  /// END SOLUTION
}


/**
     * 基于分块表示的矩阵乘法。在这种实现中，a、b和out都是适当大小的*4D*紧凑数组，
     * 例如a的尺寸为 a[m/TILE][n/TILE][TILE][TILE]。
     * 函数通过逐块进行乘法来提升性能（即本函数会调用上面实现的`AlignedDot()`）。
     *
     * 注意：本函数仅在m、n、p均为TILE的整数倍时被调用，因此可以假设除法无余数。
     *
     * 参数说明：
     *   a: 紧凑4D数组，尺寸为 m/TILE × n/TILE × TILE × TILE
     *   b: 紧凑4D数组，尺寸为 n/TILE × p/TILE × TILE × TILE
     *   out: 用于存储结果的紧凑4D数组，尺寸为 m/TILE × p/TILE × TILE × TILE
     *   m: 矩阵a和输出矩阵out的行数
     *   n: 矩阵a的列数 / 矩阵b的行数（矩阵相乘的维度匹配条件）
     *   p: 矩阵b和输出矩阵out的列数
     */
void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /// BEGIN SOLUTION
    // 初始化输出数组所有元素为0（避免残留值影响累加结果）
    for(uint32_t i=0;i<m*p;i++){
        out->ptr[i] = 0;
    }

    // 遍历输出矩阵的行块索引（i：m方向的块数 = m/TILE）
    for(uint32_t i = 0; i < m / TILE; ++i) {
        // 遍历输出矩阵的列块索引（j：p方向的块数 = p/TILE）
        for (uint32_t j = 0; j < p / TILE; ++j) {
            // 计算当前输出块在4D数组中的起始地址
            // 地址公式：(行块索引×总列块数 + 列块索引) × 单个块的元素数（TILE×TILE）
            float* c = out->ptr + (i * (p / TILE) + j) * TILE * TILE;

            // 将当前输出块初始化为0（为AlignedDot的累加操作做准备）
            memset(c, 0, TILE * TILE * sizeof(float));

            // 遍历中间维度的块索引（k：n方向的块数 = n/TILE）
            for (uint32_t k = 0; k < n / TILE; ++k) {
                // 获取输入矩阵a中当前块的起始地址
                // a的4D结构：[m块][n块][TILE行][TILE列]，地址公式：(i×n块数 + k) × 块大小
                const float* block_a = a.ptr + (i * (n / TILE) + k) * TILE * TILE;

                // 获取输入矩阵b中当前块的起始地址
                // b的4D结构：[n块][p块][TILE行][TILE列]，地址公式：(k×p块数 + j) × 块大小
                const float* block_b = b.ptr + (k * (p / TILE) + j) * TILE * TILE;

                // 调用AlignedDot进行TILE×TILE矩阵块乘法，并累加结果到当前输出块c
                AlignedDot(block_a, block_b, c);
            }
        }
    }
  /// END SOLUTION
}


/**
 * 对数组进行最大归约操作（沿指定维度取最大值）
 *
 * @param[in]  a           输入数组（待归约的原始数据）
 * @param[out] out         输出数组（归约结果，尺寸为 a.size / reduce_size）
 * @param[in]  reduce_size 每个归约组的元素数量（即沿归约维度的大小）
 *
 * 功能说明：
 *  将输入数组 a 按每组 reduce_size 个元素 划分为多个连续组，
 *  每组内取最大值作为输出数组 out 对应位置的元素。
 *  要求 a.size 必须是 reduce_size 的整数倍（out.size = a.size / reduce_size）。
 */
void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /// BEGIN SOLUTION
  // 遍历输出数组的每个位置（每个位置对应输入数组的一个归约组）
  for(size_t i = 0; i < out->size; i++){
    // 初始化当前组的最大值为组内第一个元素
    out->ptr[i] = a.ptr[i * reduce_size];

    // 遍历组内剩余元素，更新最大值
    for(size_t j = 1; j < reduce_size; j++){
      // 取当前最大值与组内第j个元素的较大值
      out->ptr[i] = std::max(out->ptr[i], a.ptr[i * reduce_size + j]);
    }
  }
  /// END SOLUTION
}

/**
 * 对数组进行求和归约操作（沿指定维度求和）
 *
 * @param[in]  a           输入数组（待归约的原始数据）
 * @param[out] out         输出数组（归约结果，尺寸为 a.size / reduce_size）
 * @param[in]  reduce_size 每个归约组的元素数量（即沿归约维度的大小）
 *
 * 功能说明：
 *  将输入数组 a 按每组 reduce_size 个元素划分为多个连续组，
 *  每组内所有元素求和作为输出数组 out 对应位置的元素。
 *  要求 a.size 必须是 reduce_size 的整数倍（out.size = a.size / reduce_size）。
 */
void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /// BEGIN SOLUTION
  // 遍历输出数组的每个位置（每个位置对应输入数组的一个归约组）
  for(size_t i = 0; i < out->size; i++){
    // 初始化当前组的和为0
    out->ptr[i] = 0;

    // 遍历组内所有元素，累加求和
    for(size_t j = 0; j < reduce_size; j++){
      out->ptr[i] += a.ptr[i * reduce_size + j];
    }
  }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

   #define REGISTER_EWISW_OP(NAME, OP) \
    m.def(NAME, [](const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
      EwiseOp(a, b, out, OP); \
    });

  #define REGISTER_SCALAR_OP(NAME, OP) \
    m.def(NAME, [](const AlignedArray& a, scalar_t val, AlignedArray* out) { \
      ScalarOp(a, val, out, OP); \
    });
  #define REGISTER_SINGLE_OP(NAME, OP) \
    m.def(NAME, [](const AlignedArray& a, AlignedArray* out) { \
      for (size_t i = 0; i < a.size; i++) { \
        out->ptr[i] = OP(a.ptr[i]); \
      } \
    });

  REGISTER_EWISW_OP("ewise_mul", std::multiplies<scalar_t>());
  REGISTER_SCALAR_OP("scalar_mul", std::multiplies<scalar_t>());
  REGISTER_EWISW_OP("ewise_div", std::divides<scalar_t>());
  REGISTER_SCALAR_OP("scalar_div", std::divides<scalar_t>());
  REGISTER_SCALAR_OP("scalar_power", static_cast<scalar_t(*)(scalar_t, scalar_t)>(std::pow));
  REGISTER_EWISW_OP("ewise_maximum", static_cast<scalar_t(*)(scalar_t, scalar_t)>(std::fmax));
  REGISTER_SCALAR_OP("scalar_maximum", static_cast<scalar_t(*)(scalar_t, scalar_t)>(std::fmax));
  REGISTER_EWISW_OP("ewise_eq", std::equal_to<scalar_t>());
  REGISTER_SCALAR_OP("scalar_eq", std::equal_to<scalar_t>());
  REGISTER_EWISW_OP("ewise_ge", std::greater_equal<scalar_t>());
  REGISTER_SCALAR_OP("scalar_ge", std::greater_equal<scalar_t>());
  REGISTER_SINGLE_OP("ewise_log", std::log);
  REGISTER_SINGLE_OP("ewise_exp", std::exp);
  REGISTER_SINGLE_OP("ewise_tanh", std::tanh);

   m.def("matmul", Matmul);
   m.def("matmul_tiled", MatmulTiled);

   m.def("reduce_max", ReduceMax);
   m.def("reduce_sum", ReduceSum);
}
