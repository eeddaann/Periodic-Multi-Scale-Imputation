import unittest
import numpy as np
import pandas as pd
from pmsi.pmsi_core import PMSIImputer

class TestPMSI(unittest.TestCase):
    def test_2d_backward_compatibility(self):
        # 1. Test instantiation with defaults
        imputer = PMSIImputer(x_size=7, y_size=7, n_evals=5)
        self.assertEqual(imputer.sizes, (7, 7))
        self.assertEqual(imputer.ndim, 2)
        self.assertEqual(imputer.pad_width, 3)

        # 2. Test kernel equivalent to the old Gaussian2DKernel
        x_std, y_std = 1.5, 2.0
        kernel_compatible = imputer._get_kernel(x_std, y_std)
        # Built-in Gaussian2DKernel
        from astropy.convolution import Gaussian2DKernel
        kernel_astropy = Gaussian2DKernel(x_stddev=x_std, y_stddev=y_std, x_size=7, y_size=7)
        self.assertTrue(np.allclose(kernel_compatible.array, kernel_astropy.array))

        # 3. Test convolve is equivalent
        img = np.random.rand(28, 24)
        img[5, 5] = np.nan
        img[10, 15] = np.nan
        res_compat = imputer._convolve(img, [y_std, x_std])
        # Old convolve logic
        padded = np.pad(img.copy(), mode='edge', pad_width=3)
        from astropy.convolution import interpolate_replace_nans
        interpolated = interpolate_replace_nans(padded, kernel_astropy)
        res_old = interpolated[3:-3, 3:-3]
        self.assertTrue(np.allclose(res_compat, res_old))

        # 4. Test fit and best_params_
        np.random.seed(42)
        good_k = ['p1', 'p2']
        binary_masks = ['m1', 'm2']
        pivot_d = {
            'p1': pd.DataFrame(np.random.rand(28, 24)),
            'p2': pd.DataFrame(np.random.rand(28, 24))
        }
        binary_d = {
            'm1': np.random.rand(28, 24) > 0.8,
            'm2': np.random.rand(28, 24) > 0.8
        }
        imputer.fit(pivot_d, binary_d, good_k, binary_masks, n_pairs_per_key=1)
        self.assertIn('x_0', imputer.best_params_)
        self.assertIn('x_1', imputer.best_params_)
        
        # Hardened assertions to check that optimal values match the original 2D TPE execution exactly
        self.assertAlmostEqual(imputer.best_params_['x_0'], 2.462922845535262)
        self.assertAlmostEqual(imputer.best_params_['x_1'], 2.7117563527545014)

        # 5. Test impute is equivalent
        res_imp = imputer.impute(img)
        res_expected = imputer._convolve(img, [imputer.best_params_['x_0'], imputer.best_params_['x_1']])
        self.assertTrue(np.allclose(res_imp, res_expected))

    def test_1d_support(self):
        imputer = PMSIImputer(sizes=(5,), n_evals=5)
        self.assertEqual(imputer.sizes, (5,))
        self.assertEqual(imputer.ndim, 1)

        # Convolve 1D
        img = np.random.rand(10)
        img[3] = np.nan
        std = [1.2]
        res = imputer._convolve(img, std)
        self.assertEqual(res.shape, (10,))
        self.assertFalse(np.isnan(res[3]))

    def test_3d_support(self):
        imputer = PMSIImputer(sizes=(3, 5, 7), n_evals=5)
        self.assertEqual(imputer.sizes, (3, 5, 7))
        self.assertEqual(imputer.ndim, 3)

        # 3D fit and optimize
        np.random.seed(42)
        good_k = ['p1']
        binary_masks = ['m1']
        pivot_d = {
            'p1': np.random.rand(4, 5, 6)
        }
        binary_d = {
            'm1': np.random.rand(4, 5, 6) > 0.8
        }
        imputer.fit(pivot_d, binary_d, good_k, binary_masks, n_pairs_per_key=1, shape=(4, 5, 6))
        self.assertIn('x_0', imputer.best_params_)
        self.assertIn('x_1', imputer.best_params_)
        self.assertIn('x_2', imputer.best_params_)

        # Impute
        miss = np.random.rand(4, 5, 6)
        miss[2, 2, 2] = np.nan
        res = imputer.impute(miss)
        self.assertEqual(res.shape, (4, 5, 6))
        self.assertFalse(np.isnan(res[2, 2, 2]))

if __name__ == '__main__':
    unittest.main()
