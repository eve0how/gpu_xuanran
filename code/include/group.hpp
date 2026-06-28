#ifndef GROUP_H
#define GROUP_H

// PA1 已有代码
#include "object3d.hpp"
#include "ray.hpp"
#include "hit.hpp"
#include <iostream>
#include <vector>


// TODO: Implement Group - add data structure to store a list of Object*
// use a vector
// Done
class Group : public Object3D {

public:

    Group() {

    }

    explicit Group (int num_objects) {
        objects.resize(num_objects);
    }

    ~Group() override {

    }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        bool anyHit = false;
        for(int i = 0; i < objects.size(); i++){
            if(objects[i] == nullptr) continue;
            if(objects[i]->intersect(r, h, tmin)){
                anyHit = true;
            }
        }
        return anyHit;
    }

    void addObject(int index, Object3D *obj) {
        objects[index] = obj; // 将对象添加到指定位置
    }

    int getGroupSize() {
        return objects.size();
    }

    Object3D *getObject(int index) const {
        return objects[index];
    }

private:
    std::vector<Object3D*> objects; // 使用vector存储Object3D指针列表

};

#endif
	
