#include "simple.h"
#include "A.h"

#include <cmath>
#include <memory>

int a = 10; 

class B {
    int b1;
    public: 
    int fb1() { return b1; }
};

int func(int a, int b) {
    int c = a + b;
    return c * c;
}

int o = func(1, 2);

int func2(int k) {
    return x + a;
}

int func3(int a) {
    return ::a * 42;
}

int func4(int k) {
    static int y = 1;
    int b = x + a;
    y += k + b + o;
    return y;
}

int func5(int k) {
    int m = func(1, 2) + func2(m);
    return m;
}

int func6(int k) {
    A obj1;
    B obj2;
    int m = obj1.f1() + A::f2() + obj2.fb1();
    return m;
}

int func7() {
    A* obj = new A();
    B* obj2 = new B();
    B** obj2_ptr = &obj2;
    return obj->f1() + (*obj2_ptr)->fb1();
}

int func8(B obj) {
    return obj.fb1() + A::f2();
}

typedef int myint;
typedef int myint2_t;
typedef myint2_t myint2;

double func9(myint m) {
    myint n = m + 1;
    myint2 l = m * 3;
    return n + m + l;
}

double func10(int m) {
    return std::cos(m);
}

int func11() {
    std::unique_ptr<A> obj(new A());
    return obj->f1();
}

template <typename T> 
int func12(T lmb) {
    return lmb(10, 12);
}

template <typename T, typename U, typename V>
int func12(T lmb, U a, V b) {
    return lmb(a, b);
}

int func12(int param) {
    return param;
}

float func12(float param) {
    return param + 1.0;
}

int func13(int x) {
    auto glambda = [=](int a, int b) { return x + a < b; };
    return func12(glambda);
}

template <typename T> 
int func14(T&& lmb) {
    return lmb(10, 12);
}

template <typename T> 
T func15(T&& lmb) {
    return lmb;
}

int main() {
    auto glambda = [](int a, int b) { return a < b; };
    func12(glambda);
    
    func12(glambda, 1, 2);

    func14(glambda);

    int const f15 = 1;
    func15(f15);

    return 0;
}