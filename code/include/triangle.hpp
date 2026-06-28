#ifndef TRIANGLE_H
#define TRIANGLE_H

// PA1 已有代码
#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

class Triangle : public Object3D {

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

	bool intersect(const Ray &ray, Hit &hit, float tmin) override {
		Vector3f rayOrig = ray.getOrigin();
        Vector3f rayDir = ray.getDirection();
		Vector3f e1 = vertices[0] - vertices[1], e2 = vertices[0] - vertices[2], s = vertices[0] - rayOrig;
		Matrix3f m1(s, e1, e2), m2(rayDir, s, e2), m3(rayDir, e1, s), m4(rayDir, e1, e2);
		float detMain = m4.determinant();
		if (fabsf(detMain) < 1e-6f) return false;
		float hitDist = m1.determinant() / detMain;
		float beta = m2.determinant() / detMain;
		float gamma = m3.determinant() / detMain;
		if(beta >= 0 && gamma >= 0 && beta + gamma <= 1){
			if(hitDist > tmin && hitDist < hit.getT()){
				hit.set(hitDist, material, normal);
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
