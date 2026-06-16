#include <bits/stdc++.h>
using namespace std;

class myqueue
{
    public:
        int *arr;
        int front, rear;
        int capacity;
        int size;
        myqueue()
        {
            arr = new int[2];
            capacity = 2;
            size = 0;
            front = rear = 0;
        }
        ~myqueue()
        {
            delete[] arr;
        }
        int getsize()const
        {
            return this->size;
        }
        int getcapacity()const{
            return this->capacity;
        }
        void push(const int &val)
        {
            if(this->size==this->capacity)
            {
                int *temp = new int[2 * capacity];
                memcpy(temp, arr, capacity*sizeof(int));
                delete[] arr;
                arr = temp;
                capacity *= 2;
            }
            rear = (rear + 1) % capacity;
            arr[rear] = val;
            size++;
        }
        void pop()
        {
            front = (front + 1) % capacity;
            size--;
        }
        void print()
        {
            for (int i = 0; i < size; i++)
            {
                cout << arr[i] << endl;
            }
            
        }
};

int main()
{
    myqueue mq;
    for (int i = 0; i <= 9;++i)
    {
        mq.push(i);
    }
    mq.print();
    return 0;
}
