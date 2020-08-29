#include <iostream>
#include <fstream>
#include <string>
#include "mpi.h"

using namespace std;


void generateMatrix(const int size,string name)
{
	int** matrix = new int*[size];
	
	for (int i = 0; i < size; i++)
	{
		matrix[i] = new int[size + 1];
		matrix[i][i] = 1;

		for (int j = 0; j < size + 1; j++)
		{
			if (i != j)
			{
				matrix[i][j] = rand() % 100;
				matrix[i][i] += matrix[i][j];
			}
		}
		if (matrix[i][i] == 0)
			matrix[i][i]++;
	}

	ofstream out1, out2;

	out1.open(name+".txt");
	out2.open("approx_" + name + ".txt");

	out1 << size << " " << size + 1 << endl;
	out2 << size << endl;

	for (auto i = 0; i < size; i++)
	{
		for (auto j = 0; j < size + 1; j++)
			out1 << matrix[i][j] << " ";
		out1 << endl;
		out2 << rand() % 10 << endl;
	}

	for (auto i = 0; i < size; i++)
		delete[] matrix[i];
	delete[] matrix;

	out1.close();
	out2.close();
	return;
}


bool FileIsExist(std::string filePath)
{
	bool isExist = false;
	filePath = filePath + ".txt";
	std::ifstream fin(filePath.c_str());

	if (fin.is_open())
		isExist = true;

	fin.close();
	return isExist;
}

int getLen(string name)
{
	int len;
	ifstream in;
	name = name + ".txt";
	in.open(name);
	if (!in.is_open()) throw std::runtime_error("error open file " + name);
	in >> len;
	in.close();
	return len;
}


void readFiles(const int size, long double* a, long double* b, long double *x, string name)
{

	cout << "Reading file " << name << endl;
	ifstream in;
	in.open(name+".txt");
	if (!in.is_open())
		throw "Can't open file " + name;

	int m, n;
	in >> m >> n;
	if (n != m + 1 || size != m)
	{
		in.close();
		throw "Incorrect matrix parameters! n = " + to_string(n) + " m = " + to_string(m);
	}

	for (auto i = 0; i < m; i++)
	{
		for (auto j = 0; j < m; j++)
			in >> a[i * m + j];
		in >> b[i];
	}
	in.close();

	name = "approx_"  +name + ".txt";
	cout << "Reading file " << name << endl;
	int m1;

	in.open(name);
	if (!in.is_open())
		throw "Can't open file " + name;

	in >> m1;
	if (m1 != m)
	{
		in.close();
		throw "Incorrect length of X vector! Global m = " + to_string(m) + " Local m = " + to_string(m1);
	}
	for (auto i = 0; i < m; i++)
		in >> x[i];

	in.close();
}

bool checkMatrix(const int m, const long double* a)
{
	for (auto i = 0; i < m; i++)
	{
		double sum = 0;
		for (auto j = 0; j < m; j++)
			if (i != j)
				sum += a[i * m + j];
		if (a[i * m + i] <= sum || a[i * m + i] == 0)
		{
			return false;
		}
	}
	return true;
}

void writeInOutputFile(const int size, const long double* x, string name)
{
	ofstream out;
	out.open(name+".txt");

	out << size << endl;
	for (auto i = 0; i < size; i++)
		out << x[i] << endl;
	out.close();
}

void sizeCalc(const int size, const int m, int * lengths, int* lengths_a, int *offsets, int* offsets_a)
{
	for (auto i = 0; i < size; i++)
		lengths[i] = m / size;
	for (auto i = 0; i < m - (m / size) * size; i++)
		lengths[i]++;
	if (size > m)
	{
		for (int i = m; i < size; i++)
			lengths[i] = 0;
	}
	for (auto i = 0; i < size; i++)
		lengths_a[i] = lengths[i] * m;

	offsets[0] = 0;
	offsets_a[0] = 0;
	for (auto i = 1; i < size; i++)
	{
		offsets[i] = offsets[i - 1] + lengths[i - 1];
		offsets_a[i] = offsets_a[i - 1] + lengths_a[i - 1];
	}
}

double findLocalMaxNorm(const int length, const int offset, const long double* x, const long double* temp_x)
{
	double norm = 0;
	for (auto i = 0; i < length; i++)
	{
		if (abs(x[offset + i] - temp_x[i]) > norm)
			norm = abs(x[offset + i] - temp_x[i]);
	}
	return norm;
}

double findGlobalMaxNorm(const int size, const long double* norms)
{
	double norm = norms[0];
	for (int i = 0; i < size; i++)
	{
		if (norms[i] > norm)
			norm = norms[i];
	}
	return norm;
}

void Jacobi(const int m, const int size, const int rank, const double eps,
	const int* offsets, const int* lengths,
	long double * temp_x, long double* temp_a, long double* temp_b,
	long double* norms, long double* x)
{
	long double norm = 1;
	auto offset = offsets[rank];
	auto length = lengths[rank];

	while (norm > eps)
	{
		for (auto i = 0; i < length; i++)
		{
			temp_x[i] = temp_b[i];

			for (auto j = 0; j < m; j++)
			{
				if (i + offset != j)
					temp_x[i] -= temp_a[i * m + j] * x[j];
			}
			temp_x[i] /= temp_a[i * m + i + offset];
		}

		norm = findLocalMaxNorm(length, offset, x, temp_x);

		MPI_Allgather(&norm, 1, MPI_LONG_DOUBLE, norms, 1, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
		MPI_Allgatherv(temp_x, length, MPI_LONG_DOUBLE, x, lengths, offsets, MPI_LONG_DOUBLE, MPI_COMM_WORLD);

		norm = findGlobalMaxNorm(size, norms);
	}
}

int main(int argc, char* argv[])
{
	long double* a = {}, *b = {}, *x = {};
	try
	{
		MPI_Init(&argc, &argv);
		int size, rank;

		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		int sizeArray = 0;            
		const double eps = 0.001;
		string nameIn;
		string nameOut;
		if (rank == 0)
		{
			cout << "Input file name" << endl;
			cin >> nameIn;
			nameIn = nameIn;

			if (!FileIsExist(nameIn))
			{
				cout << "Create New File->\nEnter size matrix:" << endl;
				cin >> sizeArray;
				generateMatrix(sizeArray, nameIn);
			}
			else
			{
				cout << "file exist->read file" << endl;
				sizeArray = getLen(nameIn);
			}
			 
			nameOut = "out_" + nameIn;
			cout << "Output file name: " << nameOut <<endl;
		}
			

		MPI_Bcast(&sizeArray, 1, MPI_LONG, 0, MPI_COMM_WORLD);

		a = new long double[sizeArray * sizeArray];
		b = new long double[sizeArray];
		x = new long double[sizeArray];

		if (rank == 0)
		{
			readFiles(sizeArray, a, b, x,nameIn);
			if (!checkMatrix(sizeArray, a))
				throw "Error";
		}


		int* lengths = new int[size];
		int* lengths_a = new int[size];
		int* offsets = new int[size];
		int* offsets_a = new int[size];
		sizeCalc(size, sizeArray, lengths, lengths_a, offsets, offsets_a);
		auto length_a = lengths_a[rank];
		auto length = lengths[rank];
		long double* temp_a = new long double[length_a];
		long double* temp_b = new long double[length];
		long double* temp_x = new long double[length];
		long double* norms = new long double[size];

		MPI_Scatterv(a, lengths_a, offsets_a, MPI_LONG_DOUBLE, temp_a, length_a, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(b, lengths, offsets, MPI_LONG_DOUBLE, temp_b, length, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(x, sizeArray, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);


		const auto start = MPI_Wtime();    
		Jacobi(sizeArray, size, rank, eps, offsets, lengths, temp_x, temp_a, temp_b, norms, x);
		const auto finish = MPI_Wtime();

		if (rank == 0)
		{
			cout << "Processes: " << size << " Size: " << sizeArray << " Time: " << finish - start << endl;
			writeInOutputFile(sizeArray, x, nameOut);
		}

		delete[] a;
		delete[] temp_a;
		delete[] b;
		delete[] temp_b;
		delete[] x;
		delete[] temp_x;
		delete[] norms;
		delete[] lengths;
		delete[] lengths_a;
		delete[] offsets;
		delete[] offsets_a;

		MPI_Finalize();
	}
	catch (string error)
	{
		delete[] a;
		delete[] b;
		delete[] x;
		cout << error;
	}

	return 0;
}