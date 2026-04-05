import { Vector, Matrix } from "../types";

/**
 * Linear Algebra Math Engine
 * No external dependencies. Pure TypeScript.
 */
export const LinearAlgebra = {
  /**
   * Computes the dot product of two vectors.
   */
  dot(v1: Vector, v2: Vector): number {
    if (v1.length !== v2.length) {
      throw new Error(`Vector dimension mismatch: ${v1.length} vs ${v2.length}`);
    }
    return v1.reduce((acc, val, i) => acc + val * v2[i], 0);
  },

  /**
   * Adds two vectors.
   */
  add(v1: Vector, v2: Vector): Vector {
    if (v1.length !== v2.length) {
      throw new Error("Vector dimension mismatch");
    }
    return v1.map((val, i) => val + v2[i]);
  },

  /**
   * Scales a vector by a constant.
   */
  scale(v: Vector, s: number): Vector {
    return v.map((val) => val * s);
  },

  /**
   * Computes the mean of a vector.
   */
  mean(v: Vector): number {
    if (v.length === 0) return 0;
    return v.reduce((acc, val) => acc + val, 0) / v.length;
  },

  /**
   * Computes the transpose of a matrix.
   */
  transpose(m: Matrix): Matrix {
    if (m.length === 0) return [];
    return m[0].map((_, i) => m.map((row) => row[i]));
  },

  /**
   * Matrix-vector multiplication.
   */
  matrixVectorMultiply(m: Matrix, v: Vector): Vector {
    if (m.length === 0 || m[0].length !== v.length) {
      throw new Error("Dimension mismatch in matrix-vector multiplication");
    }
    return m.map((row) => this.dot(row, v));
  },

  /**
   * Matrix-matrix multiplication.
   */
  matrixMultiply(m1: Matrix, m2: Matrix): Matrix {
    if (m1.length === 0 || m2.length === 0 || m1[0].length !== m2.length) {
      throw new Error("Dimension mismatch in matrix multiplication");
    }
    const result: Matrix = [];
    const m2T = this.transpose(m2);

    for (let i = 0; i < m1.length; i++) {
      result[i] = [];
      for (let j = 0; j < m2T.length; j++) {
        result[i][j] = this.dot(m1[i], m2T[j]);
      }
    }
    return result;
  },

  /**
   * Outer product: creates a matrix from two vectors as u⊗v
   */
  outerProduct(u: Vector, v: Vector): Matrix {
    return u.map((ui) => v.map((vj) => ui * vj));
  },

  /**
   * Element-wise multiplication (Hadamard product).
   */
  elementwiseMultiply(m1: Matrix | Vector, m2: Matrix | Vector): Matrix | Vector {
    const isVector = Array.isArray(m1[0]) === false;
    
    if (isVector) {
      const v1 = m1 as Vector;
      const v2 = m2 as Vector;
      if (v1.length !== v2.length) throw new Error("Vector dimension mismatch");
      return v1.map((val, i) => val * v2[i]);
    } else {
      const mat1 = m1 as Matrix;
      const mat2 = m2 as Matrix;
      if (mat1.length !== mat2.length || mat1[0].length !== mat2[0].length) {
        throw new Error("Matrix dimension mismatch");
      }
      return mat1.map((row, i) => row.map((val, j) => val * mat2[i][j]));
    }
  },

  /**
   * Computes the L2 norm (Euclidean norm) of a vector.
   */
  norm(v: Vector): number {
    return Math.sqrt(v.reduce((acc, val) => acc + val * val, 0));
  },

  /**
   * Computes the sum of all elements in a vector.
   */
  sum(v: Vector): number {
    return v.reduce((acc, val) => acc + val, 0);
  },

  /**
   * Computes the variance of a vector.
   */
  variance(v: Vector): number {
    const m = this.mean(v);
    return this.mean(v.map((val) => (val - m) ** 2));
  },

  /**
   * Computes the standard deviation of a vector.
   */
  std(v: Vector): number {
    return Math.sqrt(this.variance(v));
  },

  /**
   * Normalizes a vector to have mean 0 and std 1.
   */
  normalize(v: Vector): Vector {
    const m = this.mean(v);
    const s = this.std(v);
    if (s === 0) return v.map(() => 0);
    return v.map((val) => (val - m) / s);
  },

  /**
   * Subtracts two vectors.
   */
  subtract(v1: Vector, v2: Vector): Vector {
    if (v1.length !== v2.length) {
      throw new Error("Vector dimension mismatch");
    }
    return v1.map((val, i) => val - v2[i]);
  },

  /**
   * Element-wise power.
   */
  power(v: Vector, p: number): Vector {
    return v.map((val) => Math.pow(val, p));
  },

  /**
   * Creates a vector of zeros.
   */
  zeros(n: number): Vector {
    return Array(n).fill(0);
  },

  /**
   * Creates a vector of ones.
   */
  ones(n: number): Vector {
    return Array(n).fill(1);
  },

  /**
   * Creates a random vector in [0, 1).
   */
  random(n: number): Vector {
    return Array(n)
      .fill(0)
      .map(() => Math.random());
  },

  /**
   * Creates a random vector with normal distribution (Box-Muller).
   */
  randomGaussian(n: number, mean = 0, std = 1): Vector {
    const v: Vector = [];
    for (let i = 0; i < n; i += 2) {
      const u1 = Math.random();
      const u2 = Math.random();
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
      v.push(mean + z1 * std);
      if (v.length < n) v.push(mean + z2 * std);
    }
    return v.slice(0, n);
  },
};
