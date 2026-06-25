#ifndef CAMERA_H
#define CAMERA_H

#include "ray.hpp"
#include <vecmath.h>
#include <float.h>
#include <cmath>


class Camera {
public:
    Camera(const Vector3f &center, const Vector3f &direction, const Vector3f &up, int imgW, int imgH) {
        this->center = center;
        this->direction = direction.normalized();
        this->horizontal = Vector3f::cross(this->direction, up).normalized();
        this->up = Vector3f::cross(this->horizontal, this->direction);
        this->width = imgW;
        this->height = imgH;
    }

    // Generate rays for each screen-space coordinate
    virtual Ray generateRay(const Vector2f &point) = 0;
    virtual ~Camera() = default;

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    const Vector3f &getCenter() const { return center; }
    const Vector3f &getDirection() const { return direction; }
    const Vector3f &getHorizontal() const { return horizontal; }
    const Vector3f &getUp() const { return up; }

protected:
    // Extrinsic parameters
    Vector3f center;
    Vector3f direction;
    Vector3f up;
    Vector3f horizontal;
    // Intrinsic parameters
    int width;
    int height;
};

// TODO: Implement Perspective camera
// You can add new functions or variables whenever needed.
// Done
class PerspectiveCamera : public Camera {

public:
    PerspectiveCamera(const Vector3f &center, const Vector3f &direction,
            const Vector3f &up, int imgW, int imgH, float angle) : Camera(center, direction, up, imgW, imgH) {
        // angle is in radian
        cx = width / 2; // the center of the image plane
        cy = height / 2;
        fx = cy / tan(angle / 2); // focal length in pixel, fy = fx for square pixels
        fy = fx;
    }

    Ray generateRay(const Vector2f &point) override {
        // 
        Vector3f R_c((point[0] - cx) / fx, (cy - point[1]) / fy, 1); // ray direction in camera space
        Matrix3f R(horizontal, -up, direction); // rotation matrix
        Vector3f R_w = R * R_c;
        Vector3f O_w = center;
        return Ray(O_w, R_w);
    }

    void getProjectionParams(float &outCx, float &outCy, float &outFx, float &outFy) const {
        outCx = cx;
        outCy = cy;
        outFx = fx;
        outFy = fy;
    }

protected:
    float cx;
    float cy;
    float fx;
    float fy;
};

#endif //CAMERA_H
