"""
Test pdaqp 
"""
import unittest
import numpy
from pdaqp import MPQP

class Testing(unittest.TestCase):
    """Testing class for pdaqp."""

    def test_python_demo(self):
        H =  numpy.array([[1.5064, 0.4838], [0.4838, 1.5258]])
        f = numpy.zeros((2,1))
        F = numpy.array([[9.6652, 5.2115], [7.0732, -7.0879]])
        A = numpy.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]])
        b = 2*numpy.ones((4,1));
        B = numpy.zeros((4,2));

        thmin = -1.5*numpy.ones(2)
        thmax = 1.5*numpy.ones(2)

        # Setup mpQP and solve it 
        mpQP = MPQP(H,f,F,A,b,B,thmin,thmax)
        mpQP.solve()

        self.assertEqual(len(mpQP.CRs), 9)


if __name__ == '__main__':
    unittest.main()
