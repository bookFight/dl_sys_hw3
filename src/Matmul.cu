#define BM 3
#define BN 3
#define T 4
template <int BLOCK_DIM>
__global__ void matrixKernel1st(float *dA,float* dB,float* dC,int M,int K,int N){
//         A M*K ,B K*N C M*N 设置ceil（M/BM）Xceil（N/BN）个block，BMXBN个线程 block中一个线程对应c矩阵中的一个元素
// 通过gpu网络结构自然分块矩阵
        int row=threadIdx.x+blockIdx.x*blockDim.x;
        int col=threadIdx.y+blockIdx.y+blockDim.y;
        float tmp=0;
        if(row<M&&col<N){
            for(int s=0;s<K;s++)
                tmp=dA[row*K+s]*dB[s*N+col];
        }
        dC[row][col]=tmp;
}
__global__ void matrixKernel2nd(float *dA,float* dB,float* dC,int M,int K,int N){
// 由于计算A中每行×B中每列,会对其数据重复访问,因而可以通过share memory 来访问,share memory对当前block中的线程所有可见
//  设置 SA[BM][K],SB[K][BN] y (0,BN)
        int row=threadIdx.x+blockIdx.x*blockDim.x;
        int col=threadIdx.y+blockIdx.y+blockDim.y;
        float tmp=0;
        __share__ float SA[BM][K];
        __share__ float SA[K][BN];
        for(int id=threadIdx.y;id<K;id+=BN){
            SA[threadIdx.x][id]=dA[row*K+id];
        }
//         通过threadIdx.y 来将dA加载到SA;
        for(int id=threadIdx.x;id<K;id+=BM){
            SB[id][threadIdx.y]=dB[id*N+col];
        }
        for(int s=0;s<K;s++){
            tmp=SA[threadIdx.x][s]*SB[s][threadIdx.y];
        }
        dC[row*N+col]=tmp;
}
// 存在k过长,SA,SB存储过大的问题,再把k进行按T分段,方便代码理解,将BM,BN都设为T,由于K被分成了K/T段,所以在方法二中的计算被重复了K/T次
__global__ void matrixKernel3rd(float *dA,float* dB,float* dC,int M,int K,int N){
        int row=threadIdx.x+blockIdx.x*blockDim.x;
        int col=threadIdx.y+blockIdx.y+blockDim.y;
        float tmp=0;
        int width=(K+T-1)/T;
        __share__ float SA[T][T];
        __share__ float SB[T][T];
        for(int i=0;i<width;i++){
            if(row<M&&threadIdx.y+i*T<K)
                SA[threadIdx.x][threadIdx.y]=dA[row*K+threadIdx.y+i*T];
            }else{
                SA[threadIdx.x][threadIdx.y]=0.0f;
            }
            if(col<N&&threadIdx.x+i*T<K){
                SB[threadIdx.x][threadIdx.y]=dB[(threadIdx.x+i*T)*N+col]
            }
            __syncthreads();
            for(int s=0;s<T;s++){
                tmp+=SA[threadIdx.x][s]*SB[s][threadIdx.y];
            }
           __syncthreads();
           if(row<M&&col<N){
             dC[row*N+col]=tmp;
           }
        }
}
// 目前一个线程只计算一个元素,我们可以一个线程计算多个元素来增大访存比,原来在同一行的线程都会重复访问dA中的元素,dB同理,因而在同一线程中计算多个元素可以减少重复访问
// 但一个线程处理的元素不是越大越好,因为share memory 有限
//  M*N  BM*BN  设一个线程处理TM*TK个dA元素,TK*TN个dB元素,所以在计算时先要通过线程索引来获得dA,dB的开始索引
__global__ void matrixKernel3rd(float *dA,float* dB,float* dC,int M,int K,int N){
    __share__ float SA[BM*BK];
    __share__ float SB[BK*BN];
    int indA=TM*(threadIdx.x+blockIdx.x*blockDim.x);
    int indB=TN*(threadIdx.y+blockIdx.y*blockDim.y);
    int width=(k+BK-1)/BK;
    float tmp[TM*TN]={0.0f};
//     循环矩阵块
    for(int i=0;i<width;i++){
        for(int index_q=0;index_q<TM;index_q++){
            for(int index_k=0;index_k<TK;index_k++){
                if(indA+index_q<M&&index_k+i*BK<K){
                    SA[(threadIdx.x*TM+index_q)*BK+index_k]=dA[(indA+index_q)*K+index_k+i*BK]
                }else{
                    SA[(threadIdx.x*TM+index_q)*BK+index_k]=0.0f;
                }
            }
        }
        __syncthreads();
        for(int index_v=0;index_v<TN;index_v++){
            for(int index_k=0;index_k<TK;index_k++){
                if(indB+index_v<N&&index_k+i*BK<K){
                    SB[index_k*BN+(threadIdx.y*TN+index_v)]=dB[(indB+index_v)+(index_k+i*BK)*N]
                }else{
                    SB[index_k*BN+(threadIdx.y*TN+index_v)]=0.0f;
                }
            }
        }
        __syncthreads();
         for(int index_q=0;index_v<TM;index_q++){
            for(int index_v=0;index_v<TN;index_v++){
                for(int index_k=0;index_k<BK;index_k++){
                    tmp[index_q*TN+index_v]=SA[(threadIdx.x*TM+index_q)*BK+index_k]*SB[index_k*BN+threadIdx.y*TN+index_v];
                }
            }
        }
        __syncthreads();
        for(int index_q=0;index_v<TM;index_q++){
            for(int index_v=0;index_v<TN;index_v++){
                if(indA+index_q<M&&indB+index_v<N){
                    dC[(indA+indA)*TN+indB+index_v]=tmp[index_q*TN+index_v];
                }
            }
        }
    }
}
// 如果分块较小时,便可把tmp放入寄存器中