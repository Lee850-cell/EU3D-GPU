/*=================================================================================
 * namespace Global:  The global variables
 *				      The type number of the difference and time-advance scheme
 *=================================================================================*/
#pragma once

#include "Array.hpp"
#include <vector>
using namespace ARRAY;
#ifndef GLOBAL
#define GLOBAL

namespace Global
{
	/* Type definition of reaction scheme */
	typedef enum
	{
		Trapezoid,
		IMEX,
		DNN
	} REACT;

	/* Type definition of difference scheme */
	typedef enum
	{
		MUSCL_1,
		MUSCL_2,
		WENO
	} DIFF;

	/* Type definition of time-advance scheme */
	typedef enum
	{
		EE,
		TVD_RK3
	} ADV;

	/* Judge the input sentence */
	int SenJud(char *sentence);

	/* Split the sentence into parameters and its value */
	void ParaGet(char *sentence, char *parameter, char *value);

	/* Quick sort function */
	template <typename T>
	void swap(T &a, T &b)
	{
		T temp = a;
		a = b;
		b = temp;
	}

	template <typename T>
	int partition(std::vector<T> &arr, std::vector<T> &index, int low, int high)
	{
		T pivot = arr[high]; // 选择最后一个元素作为基准点
		int i = low - 1;	 // 初始化较小元素的索引

		for (int j = low; j < high; j++)
		{
			if (arr[j] > pivot)
			{
				i++;
				swap(arr[i], arr[j]);
				swap(index[i], index[j]);
			}
		}

		swap(arr[i + 1], arr[high]); // 将基准元素放到正确的位置
		swap(index[i + 1], index[high]);
		return i + 1;
	}

	template <typename T>
	void quickSort(std::vector<T> &arr, std::vector<T> &index, int low, int high)
	{
		if (low < high)
		{
			// 找到分区点
			int pi = partition(arr, index, low, high);

			// 对分区点左边和右边的子数组进行递归排序
			quickSort(arr, index, low, pi - 1);
			quickSort(arr, index, pi + 1, high);
		}
	}

	template <typename T>
	int findIndex(std::vector<T> &index, int target)
	{
		for (int i = 0; i < index.GetSize(); i++)
		{
			if (index[i] == target)
			{
				return i; // 找到目标数字，返回索引
			}
		}

		return -1; // 目标数字不存在于数组中，返回-1表示未找到
	}

	/* Pow for integer mi */
	template <typename T>
	T Power(T a, size_t b)
	{
		T ans = 1;
		for (size_t i = 0; i < b; i++)
			ans *= a;
		return ans;
	}
}
#endif

extern int myid;
extern int numprocs;
extern int m_block_x, m_block_y, m_block_z;
extern int m_left, m_right, m_up, m_down, m_front, m_back;
extern int myid_x, myid_y, myid_z;
extern int num_thread;
