import unittest
import math
import sys
import os

# Добавляем корень проекта в sys.path, чтобы импорт работал
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unimathutils.linalg.vector import Vector 
from unimathutils.linalg.matrix import Matrix

from unimathutils.stats.combinatorics import (
    factorial,
    combinations,
    permutations,
    arrangements,
    arrangements_with_repetition,
    binomial_probability
)


class TestVector(unittest.TestCase):

    def setUp(self):
        #вектор для тестов
        self.v1 = Vector([1, 2, 3])
        self.v2 = Vector([4, 5, 6])
        self.v_float = Vector([1.5, 2.5])

    # Тест инициализации
    def test_init(self):
        # Проверка длины
        self.assertEqual(len(self.v1), 3)
        # Проверка доступа по индексу
        self.assertEqual(self.v1[0], 1)
        # Проверка защиты от неправильного типа
        with self.assertRaises(TypeError):
            Vector(["a", "b"])

    # Тест арифметики
    def test_add_sub_mul(self):
        # Сложение
        v_sum = self.v1 + self.v2
        self.assertEqual(v_sum.data, [5, 7, 9])
        # Вычитание
        v_sub = self.v2 - self.v1
        self.assertEqual(v_sub.data, [3, 3, 3])
        # Умножение на скаляр



class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.m1 = Matrix(2,2, [1, 2, 3, 6])
        self.m2 = Matrix(2,3,[1,2,3,4,5, 6])

        self.m3 = Matrix(2, 2, [5, 6, 7, 8])
        self.m4 = Matrix(2,2, [1,2,3,4])

        # Вектор длины 2
        self.v = Vector([1, 1])

        #Инициализация

    def test_init(self):
        self.assertEqual(self.m1.rows, 2)
        self.assertEqual(self.m1.cols, 2)
        self.assertEqual(self.m1.data, [1, 2, 3, 6])



    def test_get_item(self):
        self.assertEqual(self.m1[0, 0], 1)
        self.assertEqual(self.m1[0, 1], 2)
        self.assertEqual(self.m1[1, 0], 3)
        self.assertEqual(self.m1[1, 1], 6)

    def test_set_item(self):
        self.m1[0, 0] = 100
        self.assertEqual(self.m1[0, 0], 100)



    def test_inheritance_from_vector(self):
        self.assertIsInstance(self.m1, Vector)

    def test_add_matrices(self):
        res = self.m1 + self.m3
        self.assertIsInstance(res, Matrix)
        self.assertEqual(res.data, [6, 8, 10, 14])


    def test_scalar_multiplication(self):
        res = self.m1 * 2
        self.assertEqual(res.data, [2, 4, 6, 12])

        ## Матричное умножение

    def test_matrix_matrix_multiplication(self):
        # [1 2]   [5 6]   [19 22]
        # [3 6] x [7 8] = [43 50]
        res = self.m4 @ self.m3
        self.assertIsInstance(res, Matrix)
        self.assertEqual(res.data, [19, 22, 43, 50])

    def test_matrix_vector_multiplication(self):
        # [1 2] * [1] = [3]
        # [3 6]   [1]   [7]
        res = self.m1 @ self.v
        self.assertIsInstance(res, Vector)
        self.assertNotIsInstance(res, Matrix)
        self.assertEqual(res.data, [3, 9])

    def test_matmul_wrong_dims(self):
        m = Matrix(3, 3)
        with self.assertRaises(ValueError):
            _ = self.m1 @ m

    def test_matmul_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = self.m1 @ 5


#########
#########
#########
class TestFactorial(unittest.TestCase):

    def test_factorial_basic(self):
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)
        self.assertEqual(factorial(5), 120)


    def test_factorial_wrong_type(self):
        with self.assertRaises(TypeError):
            factorial(2.5)


class TestCombinations(unittest.TestCase):

    def test_combinations_basic(self):
        self.assertEqual(combinations(5, 2), 10)
        self.assertEqual(combinations(6, 0), 1)

    def test_combinations_invalid(self):
        with self.assertRaises(ValueError):
            combinations(5, 7)


class TestPermutations(unittest.TestCase):

    def test_permutations_basic(self):
        self.assertEqual(permutations(0), 1)
        self.assertEqual(permutations(4), 24)

    def test_permutations_negative(self):
        with self.assertRaises(ValueError):
            permutations(-3)


class TestArrangements(unittest.TestCase):

    def test_arrangements_basic(self):
        # A(5, 2) = 5 * 4 = 20
        self.assertEqual(arrangements(5, 2), 20)

    def test_arrangements_edge_cases(self):
        self.assertEqual(arrangements(5, 0), 1)
        self.assertEqual(arrangements(5, 5), factorial(5))

    def test_arrangements_invalid(self):
        with self.assertRaises(ValueError):
            arrangements(3, 5)


class TestArrangementsWithRepetition(unittest.TestCase):

    def test_arrangements_with_repetition_basic(self):
        self.assertEqual(arrangements_with_repetition(3, 2), 9)
        self.assertEqual(arrangements_with_repetition(5, 0), 1)

    def test_arrangements_with_repetition_invalid(self):
        with self.assertRaises(ValueError):
            arrangements_with_repetition(-1, 2)
        with self.assertRaises(ValueError):
            arrangements_with_repetition(3, -2)


class TestBinomialProbability(unittest.TestCase):

    def test_binomial_probability_basic(self):
        # P(X=2) for n=4, p=0.5

        self.assertAlmostEqual(
            binomial_probability(4, 2, 0.5),
            0.375
        )

    def test_binomial_probability_edges(self):
        self.assertEqual(binomial_probability(5, 0, 0.3), (1 - 0.3) ** 5)
        self.assertEqual(binomial_probability(5, 5, 0.3), 0.3 ** 5)

    def test_binomial_probability_invalid_p(self):
        with self.assertRaises(ValueError):
            binomial_probability(5, 2, -0.1)
        with self.assertRaises(ValueError):
            binomial_probability(5, 2, 1.5)

    def test_binomial_probability_invalid_k(self):
        with self.assertRaises(ValueError):
            binomial_probability(5, 6, 0.5)
