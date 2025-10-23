#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <ostream>

namespace bls {

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double norm() const { return std::sqrt(x * x + y * y + z * z); }
    double squaredNorm() const { return x * x + y * y + z * z; }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vec3& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vec3& operator/=(double scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
};

inline Vec3 operator+(Vec3 lhs, const Vec3& rhs) {
    lhs += rhs;
    return lhs;
}

inline Vec3 operator-(Vec3 lhs, const Vec3& rhs) {
    lhs -= rhs;
    return lhs;
}

inline Vec3 operator*(Vec3 lhs, double scalar) {
    lhs *= scalar;
    return lhs;
}

inline Vec3 operator*(double scalar, Vec3 rhs) {
    rhs *= scalar;
    return rhs;
}

inline Vec3 operator/(Vec3 lhs, double scalar) {
    lhs /= scalar;
    return lhs;
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

struct Mat3 {
    // column-major storage to align with molecular simulations convention.
    std::array<Vec3, 3> cols{};

    Mat3() = default;
    Mat3(const Vec3& c0, const Vec3& c1, const Vec3& c2) {
        cols[0] = c0;
        cols[1] = c1;
        cols[2] = c2;
    }

    const Vec3& operator[](std::size_t idx) const { return cols[idx]; }
    Vec3& operator[](std::size_t idx) { return cols[idx]; }

    double determinant() const {
        return dot(cols[0], cross(cols[1], cols[2]));
    }
};

inline Vec3 operator*(const Mat3& m, const Vec3& v) {
    return Vec3{
        m.cols[0].x * v.x + m.cols[1].x * v.y + m.cols[2].x * v.z,
        m.cols[0].y * v.x + m.cols[1].y * v.y + m.cols[2].y * v.z,
        m.cols[0].z * v.x + m.cols[1].z * v.y + m.cols[2].z * v.z};
}

inline Mat3 operator*(const Mat3& m, double scalar) {
    return Mat3{m.cols[0] * scalar, m.cols[1] * scalar, m.cols[2] * scalar};
}

inline Mat3 operator*(double scalar, const Mat3& m) {
    return m * scalar;
}

inline Mat3 transpose(const Mat3& m) {
    return Mat3{
        Vec3{m.cols[0].x, m.cols[1].x, m.cols[2].x},
        Vec3{m.cols[0].y, m.cols[1].y, m.cols[2].y},
        Vec3{m.cols[0].z, m.cols[1].z, m.cols[2].z}};
}

inline Mat3 inverse(const Mat3& m) {
    Vec3 c0 = m.cols[0];
    Vec3 c1 = m.cols[1];
    Vec3 c2 = m.cols[2];

    Vec3 r0 = cross(c1, c2);
    Vec3 r1 = cross(c2, c0);
    Vec3 r2 = cross(c0, c1);

    double det = dot(c0, r0);
    if (std::fabs(det) < 1e-12) {
        return Mat3{};
    }
    double inv_det = 1.0 / det;

    Mat3 inv;
    inv.cols[0] = r0 * inv_det;
    inv.cols[1] = r1 * inv_det;
    inv.cols[2] = r2 * inv_det;
    return transpose(inv);
}

inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

inline double maxComponent(const Vec3& v) {
    return std::max({std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)});
}

inline Vec3 roundVec(const Vec3& v) {
    return Vec3{std::round(v.x), std::round(v.y), std::round(v.z)};
}

}  // namespace bls

