#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>
#include <iostream>
using namespace std;

// TODO: implement this class and add more fields as necessary,
// Done
class Triangle: public Object3D {

public:
	Triangle() = delete;

    // a b c are three vertex positions of the triangle
	Triangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, Material* m) : Object3D(m) {
		this->vertices[0] = a;
		this->vertices[1] = b;
		this->vertices[2] = c;
		normal = Vector3f::cross(b - a, c - a);
		normal.normalize();
	}

	bool intersect( const Ray& ray,  Hit& hit , float tmin) override {
		Vector3f R_0 = ray.getOrigin();
        Vector3f R_d = ray.getDirection();
		Vector3f e1 = vertices[0] - vertices[1], e2 = vertices[0] - vertices[2], s = vertices[0] - R_0;
		Matrix3f m1(s, e1, e2), m2(R_d, s, e2), m3(R_d, e1, s), m4(R_d, e1, e2);
		float chu = m4.determinant(); // 按照PPT中的方法实现
		if (fabs(chu) < 1e-6) return false;
		float t = m1.determinant() / chu;
		float beta = m2.determinant() / chu;
		float gamma = m3.determinant() / chu;
		if(beta >= 0 && gamma >= 0 && beta + gamma <= 1){
			if(t > tmin && t < hit.getT()){
				hit.set(t, material, normal);
				return true;
			}
		} 
        return false;
	}
	Vector3f normal;
	Vector3f vertices[3];
protected:

};

#endif //TRIANGLE_H
