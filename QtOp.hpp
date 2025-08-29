#ifndef QALGEBRA_HPP
#define QALGEBRA_HPP

#include <armadillo>
#include <complex>
#include <iostream>

class QVecs {
public:
    arma::cx_mat data;

    QVecs() = default;
    QVecs(const arma::cx_mat& d) : data(d) {}

    // Pretty printing
    friend std::ostream& operator<<(std::ostream& os, const QVecs& q) {
        os << q.data;
        return os;
    }

    // Scalar ops
    QVecs operator*(std::complex<double> s) const {
        return QVecs(data * s);
    }

    QVecs operator/(std::complex<double> s) const {
        return QVecs(data / s);
    }

    QVecs operator+(const QVecs& other) const {
        return QVecs(data + other.data);
    }

    QVecs operator-(const QVecs& other) const {
        return QVecs(data - other.data);
    }

    bool operator==(const QVecs& other) const {
        return arma::approx_equal(data, other.data, "absdiff", 1e-10);
    }

    QVecs pow(int x) const {
        return QVecs(arma::pow(data, x));
    }

    arma::cx_mat CT() const {
        return data.t();
    }
};


class Ket : public QVecs {
public:
    Ket(const arma::cx_vec& v) : QVecs(arma::reshape(v, v.n_rows, 1)) {}

    class Bra dagger() const;
    arma::cx_mat operator%(const class Bra& b) const;
    class Bra CT() const;
};


class Bra : public QVecs {
public:
    Bra(const arma::cx_vec& v) : QVecs(arma::reshape(v.t(), 1, v.n_rows)) {}

    Ket dagger() const {
        return Ket(data.t());
    }

    std::complex<double> operator%(const Ket& k) const {
        return arma::accu(data % k.data);
    }

    Ket CT() const {
        return Ket(data.t());
    }
};


inline Bra Ket::dagger() const {
    return Bra(arma::vectorise(data));
}

inline arma::cx_mat Ket::operator%(const Bra& b) const {
    return data * b.data;
}

inline Bra Ket::CT() const {
    return Bra(arma::vectorise(data));
}


class QOp {
public:
    arma::cx_mat data;

    QOp() = default;
    QOp(const arma::cx_mat& d) : data(d) {}

    friend std::ostream& operator<<(std::ostream& os, const QOp& q) {
        os << q.data;
        return os;
    }

    bool operator==(const QOp& other) const {
        return arma::approx_equal(data, other.data, "absdiff", 1e-10);
    }

    QOp operator+(const QOp& other) const {
        return QOp(data + other.data);
    }

    QOp operator-(const QOp& other) const {
        return QOp(data - other.data);
    }

    QVecs operator%(const Ket& k) const {
        return QVecs(data * k.data);
    }

    QVecs operator%(const Bra& b) const {
        return QVecs(b.data * data);
    }

    QOp operator%(const QOp& other) const {
        return QOp(data * other.data);
    }

    QOp operator*(std::complex<double> s) const {
        return QOp(data * s);
    }

    bool isHermitian() const {
        return arma::approx_equal(data, arma::conj(data.t()), "absdiff", 1e-10);
    }

    QOp dagger() const {
        return QOp(arma::conj(data.t()));
    }
};


class Op : public QOp {
public:
    Op(const arma::cx_mat& d) : QOp(d) {}

    bool isUnitary() const {
        QOp x = (*this) % this->dagger();
        Op I = Op(arma::eye<arma::cx_mat>(data.n_rows, data.n_cols));
        return x == I;
    }
};


// commutators
inline bool commute(const Op& A, const Op& B) {
    return arma::approx_equal((A%B).data, (B%A).data, "absdiff", 1e-10);
}

inline Op commutator(const Op& A, const Op& B) {
    return Op((A%B).data - (B%A).data);
}

#endif // QALGEBRA_HPP
