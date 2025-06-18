#include "PullMethod.h"

void PullMethod::memory()
{
	cudaMalloc((void**)&dir_, sizeof(V3));
}

void PullMethod::calcDirection(Prop* prop_)
{
    CalcDirection << <1, 100 >> > (prop_, *this);
}

void PullMethod::addPullForce(Prop* prop_)
{
    AddPullForce << <1, 100 >> > (prop_, *this);
}

__global__ void AddPullForce(Prop* prop_, PullMethod pobject_) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < 100) {
        for (int i = 0; i < DIMSIZE; i++) {
            prop_[tid].force[i] += (-pobject_.pf_ * pobject_.dir_[0].e[i]);
        }
    }
}

__global__ void CalcDirection(Prop* prop_, PullMethod pobject_)
{
	int tid = threadIdx.x;
	__shared__ float sharedData[3*100];

	if (tid < 100) {
		for (int i = 0; i < DIMSIZE; i++) {
			sharedData[tid*3+i] = prop_[sim::totalnumber_ - tid - 1].pos[i] - prop_[tid].pos[i];
		}

		__syncthreads();

        // �����ֵ��ƽ��
        if (tid == 0) {
            float avgDifference[3] = { 0.0f, 0.0f, 0.0f }; // �洢ƽ����ֵ

            // �ۼ����еĲ�ֵ
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < DIMSIZE; j++) {
                    avgDifference[j] += sharedData[i * 3 + j]; // �ۼӲ�ֵ
                }
            }

            // ����ƽ����ֵ
            for (int j = 0; j < DIMSIZE; j++) {
                avgDifference[j] /= 100.0f; // ����ƽ��
            }

            // ���㵥λ����
            float magnitude = 0.0f;
            for (int j = 0; j < DIMSIZE; j++) {
                magnitude += avgDifference[j] * avgDifference[j]; // ����ģ��ƽ��
            }
            magnitude = sqrtf(magnitude); // ����ģ

            // ��һ��Ϊ��λ����
            float unitVector[3];
            if (magnitude > 0) { // ��ֹ������
                for (int j = 0; j < DIMSIZE; j++) {
                    pobject_.dir_[0].e[j] = avgDifference[j] / magnitude; // ��һ��
                }
                //printf("%f %f %f\n", pobject_.dir_[0].e[0], pobject_.dir_[0].e[1], pobject_.dir_[0].e[2]);
            }
            else {
                // ���ģΪ�㣬�������� unitVector Ϊ������
                for (int j = 0; j < DIMSIZE; j++) {
                    pobject_.dir_[0].e[j] = 0.0f;
                }
            }
            // ��������Խ� avgDifference ��������ں�������
            // ���磬���Խ���д��ȫ���ڴ�
        }
	}
}

