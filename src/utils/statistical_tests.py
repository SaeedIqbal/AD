import numpy as np
import logging
from typing import Tuple, Optional, Union, List, Dict, Any
from scipy import stats
from scipy.stats import chi2, f, mannwhitneyu, kruskal
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, pinv

# Configure module-specific logger
logger = logging.getLogger(__name__)


class StatisticalError(Exception):
    """Custom exception for statistical testing errors."""
    pass


class ConditionalIndependenceTester:
    """
    Conditional Independence Testing for Nonstationary Modulation Detection.
    
    This class implements rigorous statistical tests for conditional independence
    between brain dynamics and auxiliary physiological signals, controlling for
    brain's short-term history. It supports both permutation-based and 
    asymptotic approaches as required by the modulation detection framework.
    """
    
    def __init__(
        self,
        test_method: str = 'permutation',
        significance_alpha: float = 0.05,
        permutation_count: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize conditional independence tester.
        
        Parameters
        ----------
        test_method : str, optional
            Testing method ('permutation' or 'asymptotic', default: 'permutation').
        significance_alpha : float, optional
            Significance level (default: 0.05).
        permutation_count : int, optional
            Number of permutations for permutation test (default: 1000).
        random_state : int, optional
            Random seed for reproducibility (default: 42).
        
        Raises
        ------
        StatisticalError
            If input parameters are invalid.
        """
        if test_method not in ['permutation', 'asymptotic']:
            raise StatisticalError("test_method must be 'permutation' or 'asymptotic'.")
        if not (0 < significance_alpha < 1):
            raise StatisticalError("significance_alpha must be in (0, 1).")
        if permutation_count <= 0:
            raise StatisticalError("permutation_count must be positive.")
        
        self.test_method = test_method
        self.significance_alpha = float(significance_alpha)
        self.permutation_count = int(permutation_count)
        self.random_state = int(random_state)
        np.random.seed(self.random_state)
    
    def _validate_conditional_test_inputs(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> None:
        """Validate inputs for conditional independence test."""
        if not all(isinstance(arr, np.ndarray) for arr in [x, y, z]):
            raise StatisticalError("All inputs must be NumPy arrays.")
        if x.ndim != 2 or y.ndim != 1 or z.ndim != 2:
            raise StatisticalError("x and z must be 2D, y must be 1D.")
        if not (x.shape[0] == y.shape[0] == z.shape[0]):
            raise StatisticalError("All arrays must have same number of samples.")
        if not all(np.all(np.isfinite(arr)) for arr in [x, y, z]):
            raise StatisticalError("Input arrays contain non-finite values.")
    
    def _partial_correlation_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute partial correlation test statistic and p-value.
        
        Parameters
        ----------
        x : np.ndarray
            Primary variable (brain state), shape (n_samples, n_features).
        y : np.ndarray
            Target variable (auxiliary signal), shape (n_samples,).
        z : np.ndarray
            Conditioning variable (brain history), shape (n_samples, n_cond_features).
        
        Returns
        -------
        tuple
            (test_statistic, p_value)
        """
        try:
            n_samples = x.shape[0]
            n_x_features = x.shape[1]
            n_z_features = z.shape[1]
            
            # Residualize x and y against z using least squares
            # x_residual = x - z @ (z^T z)^(-1) z^T x
            ztz = z.T @ z
            ztz_reg = ztz + np.eye(n_z_features) * 1e-12  # regularization
            ztz_inv = pinv(ztz_reg)
            
            # Residualize x (multivariate)
            beta_x = ztz_inv @ z.T @ x
            x_residual = x - z @ beta_x
            
            # Residualize y (univariate)
            beta_y = ztz_inv @ z.T @ y
            y_residual = y - z @ beta_y
            
            # Compute partial correlation
            # For multivariate x, use canonical correlation or average correlation
            if n_x_features == 1:
                # Univariate case
                numerator = np.sum(x_residual.flatten() * y_residual)
                denominator = np.sqrt(np.sum(x_residual**2) * np.sum(y_residual**2))
                if denominator == 0:
                    partial_corr = 0.0
                else:
                    partial_corr = numerator / denominator
            else:
                # Multivariate case - use first canonical correlation
                # Simplified: average correlation across features
                correlations = []
                for i in range(n_x_features):
                    num = np.sum(x_residual[:, i] * y_residual)
                    den = np.sqrt(np.sum(x_residual[:, i]**2) * np.sum(y_residual**2))
                    if den > 0:
                        correlations.append(num / den)
                if correlations:
                    partial_corr = np.mean(correlations)
                else:
                    partial_corr = 0.0
            
            # Compute test statistic (F-statistic approximation)
            df1 = 1  # numerator degrees of freedom
            df2 = n_samples - n_z_features - 2  # denominator degrees of freedom
            if df2 <= 0:
                return 0.0, 1.0
            
            if abs(partial_corr) >= 1.0:
                partial_corr = np.sign(partial_corr) * 0.999999
            
            f_stat = (partial_corr**2 / df1) / ((1 - partial_corr**2) / df2)
            p_value = 1.0 - f.cdf(f_stat, df1, df2)
            
            return float(f_stat), float(p_value)
            
        except Exception as e:
            raise StatisticalError(f"Partial correlation test failed: {str(e)}")
    
    def _conditional_permutation_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform conditional permutation test for independence.
        
        Parameters
        ----------
        x : np.ndarray
            Primary variable, shape (n_samples, n_features).
        y : np.ndarray
            Target variable, shape (n_samples,).
        z : np.ndarray
            Conditioning variable, shape (n_samples, n_cond_features).
        
        Returns
        -------
        tuple
            (observed_statistic, p_value)
        """
        try:
            # Compute observed test statistic
            observed_stat, _ = self._partial_correlation_test(x, y, z)
            
            # Generate null distribution via conditional permutation
            null_stats = []
            y_shuffled = y.copy()
            
            for _ in range(self.permutation_count):
                # Shuffle y while preserving its relationship with z
                # Simple approach: shuffle within bins of similar z values
                # More sophisticated: use conditional permutation via regression residuals
                
                # Residualize y against z
                ztz = z.T @ z
                ztz_reg = ztz + np.eye(z.shape[1]) * 1e-12
                ztz_inv = pinv(ztz_reg)
                beta_y = ztz_inv @ z.T @ y
                y_residual = y - z @ beta_y
                
                # Shuffle residuals
                np.random.shuffle(y_residual)
                
                # Add back conditional mean
                y_shuffled = z @ beta_y + y_residual
                
                # Compute test statistic for shuffled data
                null_stat, _ = self._partial_correlation_test(x, y_shuffled, z)
                null_stats.append(null_stat)
            
            # Compute p-value
            null_stats = np.array(null_stats)
            p_value = np.mean(null_stats >= observed_stat)
            
            return float(observed_stat), float(p_value)
            
        except Exception as e:
            raise StatisticalError(f"Conditional permutation test failed: {str(e)}")
    
    def test_conditional_independence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Test conditional independence H0: x ⊥ y | z.
        
        Parameters
        ----------
        x : np.ndarray
            Primary variable (brain state).
        y : np.ndarray
            Target variable (auxiliary signal).
        z : np.ndarray
            Conditioning variable (brain history).
        
        Returns
        -------
        tuple
            (test_statistic, p_value, is_significant)
            - test_statistic : float - Test statistic value
            - p_value : float - p-value for the test
            - is_significant : bool - Whether to reject H0 at alpha level
        """
        try:
            self._validate_conditional_test_inputs(x, y, z)
            
            if self.test_method == 'permutation':
                test_stat, p_val = self._conditional_permutation_test(x, y, z)
            else:  # asymptotic
                test_stat, p_val = self._partial_correlation_test(x, y, z)
            
            is_significant = p_val < self.significance_alpha
            return float(test_stat), float(p_val), bool(is_significant)
            
        except Exception as e:
            if isinstance(e, StatisticalError):
                raise
            raise StatisticalError(f"Conditional independence test failed: {str(e)}")


class QuantizationSensitivityTester:
    """
    Statistical Testing for Quantization Sensitivity Analysis.
    
    This class provides statistical tests to evaluate whether quantization 
    sensitivity differs significantly between groups (e.g., AD vs FTD), 
    supporting the interpretation of representation artifacts in AD.
    """
    
    def __init__(self, significance_alpha: float = 0.05):
        """
        Initialize quantization sensitivity tester.
        
        Parameters
        ----------
        significance_alpha : float, optional
            Significance level (default: 0.05).
        """
        if not (0 < significance_alpha < 1):
            raise StatisticalError("significance_alpha must be in (0, 1).")
        self.significance_alpha = float(significance_alpha)
    
    def _validate_group_comparison_inputs(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> None:
        """Validate inputs for group comparison."""
        if not all(isinstance(arr, np.ndarray) for arr in [group1, group2]):
            raise StatisticalError("Groups must be NumPy arrays.")
        if group1.ndim != 1 or group2.ndim != 1:
            raise StatisticalError("Groups must be 1D arrays.")
        if len(group1) == 0 or len(group2) == 0:
            raise StatisticalError("Groups cannot be empty.")
        if not all(np.all(np.isfinite(arr)) for arr in [group1, group2]):
            raise StatisticalError("Group arrays contain non-finite values.")
    
    def mann_whitney_u_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Perform Mann-Whitney U test for quantization sensitivity comparison.
        
        Parameters
        ----------
        group1 : np.ndarray
            Quantization sensitivity values for group 1.
        group2 : np.ndarray
            Quantization sensitivity values for group 2.
        
        Returns
        -------
        tuple
            (u_statistic, p_value, is_significant)
        """
        try:
            self._validate_group_comparison_inputs(group1, group2)
            
            # Perform Mann-Whitney U test
            u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
            
            is_significant = p_val < self.significance_alpha
            return float(u_stat), float(p_val), bool(is_significant)
            
        except Exception as e:
            raise StatisticalError(f"Mann-Whitney U test failed: {str(e)}")
    
    def kruskal_wallis_test(
        self,
        *groups: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Perform Kruskal-Wallis H test for multiple group comparison.
        
        Parameters
        ----------
        *groups : np.ndarray
            Variable number of group arrays.
        
        Returns
        -------
        tuple
            (h_statistic, p_value, is_significant)
        """
        try:
            if len(groups) < 2:
                raise StatisticalError("At least two groups required.")
            
            for group in groups:
                if not isinstance(group, np.ndarray) or group.ndim != 1:
                    raise StatisticalError("All groups must be 1D NumPy arrays.")
                if len(group) == 0:
                    raise StatisticalError("Groups cannot be empty.")
                if not np.all(np.isfinite(group)):
                    raise StatisticalError("Group arrays contain non-finite values.")
            
            # Perform Kruskal-Wallis test
            h_stat, p_val = kruskal(*groups)
            
            is_significant = p_val < self.significance_alpha
            return float(h_stat), float(p_val), bool(is_significant)
            
        except Exception as e:
            raise StatisticalError(f"Kruskal-Wallis test failed: {str(e)}")


class ClinicalCorrelationAnalyzer:
    """
    Clinical Correlation Analysis for Mechanism Validation.
    
    This class implements correlation analyses between mechanism-specific 
    metrics and clinical measures (e.g., MMSE), with appropriate statistical
    testing and multiple comparison correction.
    """
    
    def __init__(
        self,
        correlation_method: str = 'spearman',
        significance_alpha: float = 0.05,
        multiple_comparison_correction: str = 'fdr'
    ):
        """
        Initialize clinical correlation analyzer.
        
        Parameters
        ----------
        correlation_method : str, optional
            Correlation method ('pearson' or 'spearman', default: 'spearman').
        significance_alpha : float, optional
            Significance level (default: 0.05).
        multiple_comparison_correction : str, optional
            Correction method ('fdr', 'bonferroni', or 'none', default: 'fdr').
        """
        if correlation_method not in ['pearson', 'spearman']:
            raise StatisticalError("correlation_method must be 'pearson' or 'spearman'.")
        if not (0 < significance_alpha < 1):
            raise StatisticalError("significance_alpha must be in (0, 1).")
        if multiple_comparison_correction not in ['fdr', 'bonferroni', 'none']:
            raise StatisticalError("multiple_comparison_correction must be 'fdr', 'bonferroni', or 'none'.")
        
        self.correlation_method = correlation_method
        self.significance_alpha = float(significance_alpha)
        self.multiple_comparison_correction = multiple_comparison_correction
    
    def _validate_correlation_inputs(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> None:
        """Validate inputs for correlation analysis."""
        if not all(isinstance(arr, np.ndarray) for arr in [x, y]):
            raise StatisticalError("Inputs must be NumPy arrays.")
        if x.ndim != 1 or y.ndim != 1:
            raise StatisticalError("Inputs must be 1D arrays.")
        if len(x) != len(y):
            raise StatisticalError("Input arrays must have same length.")
        if len(x) < 3:
            raise StatisticalError("Need at least 3 samples for correlation.")
        if not all(np.all(np.isfinite(arr)) for arr in [x, y]):
            raise StatisticalError("Input arrays contain non-finite values.")
    
    def _compute_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """Compute correlation coefficient and p-value."""
        try:
            if self.correlation_method == 'pearson':
                corr_coef, p_val = stats.pearsonr(x, y)
            else:  # spearman
                corr_coef, p_val = stats.spearmanr(x, y)
            
            return float(corr_coef), float(p_val)
            
        except Exception as e:
            raise StatisticalError(f"Correlation computation failed: {str(e)}")
    
    def _apply_multiple_comparison_correction(
        self,
        p_values: np.ndarray
    ) -> np.ndarray:
        """Apply multiple comparison correction to p-values."""
        try:
            if self.multiple_comparison_correction == 'none':
                return p_values
            elif self.multiple_comparison_correction == 'bonferroni':
                corrected_p = np.minimum(p_values * len(p_values), 1.0)
                return corrected_p
            else:  # fdr (Benjamini-Hochberg)
                sorted_indices = np.argsort(p_values)
                sorted_p = p_values[sorted_indices]
                m = len(p_values)
                corrected_p = np.zeros_like(p_values)
                
                for i, p in enumerate(sorted_p):
                    corrected_p[sorted_indices[i]] = min(p * m / (i + 1), 1.0)
                
                # Ensure monotonicity
                for i in range(m - 2, -1, -1):
                    corrected_p[sorted_indices[i]] = min(
                        corrected_p[sorted_indices[i]], 
                        corrected_p[sorted_indices[i + 1]]
                    )
                
                return corrected_p
                
        except Exception as e:
            raise StatisticalError(f"Multiple comparison correction failed: {str(e)}")
    
    def analyze_correlation(
        self,
        clinical_scores: np.ndarray,
        mechanism_metrics: np.ndarray,
        group_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float, bool, float]:
        """
        Analyze correlation between clinical scores and mechanism metrics.
        
        Parameters
        ----------
        clinical_scores : np.ndarray
            Clinical scores (e.g., MMSE), shape (n_subjects,).
        mechanism_metrics : np.ndarray
            Mechanism-specific metrics, shape (n_subjects,).
        group_mask : np.ndarray, optional
            Boolean mask for subgroup analysis, shape (n_subjects,).
        
        Returns
        -------
        tuple
            (correlation_coefficient, p_value, is_significant, corrected_p_value)
        """
        try:
            self._validate_correlation_inputs(clinical_scores, mechanism_metrics)
            
            # Apply group mask if provided
            if group_mask is not None:
                if group_mask.dtype != bool:
                    raise StatisticalError("group_mask must be boolean.")
                if len(group_mask) != len(clinical_scores):
                    raise StatisticalError("group_mask length mismatch.")
                clinical_scores = clinical_scores[group_mask]
                mechanism_metrics = mechanism_metrics[group_mask]
            
            # Compute correlation
            corr_coef, p_val = self._compute_correlation(clinical_scores, mechanism_metrics)
            
            # For single correlation, corrected p-value is same as original
            corrected_p_val = p_val
            
            is_significant = p_val < self.significance_alpha
            return float(corr_coef), float(p_val), bool(is_significant), float(corrected_p_val)
            
        except Exception as e:
            if isinstance(e, StatisticalError):
                raise
            raise StatisticalError(f"Clinical correlation analysis failed: {str(e)}")
    
    def analyze_multiple_correlations(
        self,
        clinical_scores: np.ndarray,
        metric_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze multiple correlations with correction for multiple comparisons.
        
        Parameters
        ----------
        clinical_scores : np.ndarray
            Clinical scores, shape (n_subjects,).
        metric_dict : dict
            Dictionary mapping metric names to metric arrays.
        
        Returns
        -------
        dict
            Dictionary of correlation results for each metric.
        """
        try:
            if len(metric_dict) == 0:
                raise StatisticalError("metric_dict cannot be empty.")
            
            # Compute correlations for all metrics
            correlations = {}
            p_values = []
            metric_names = list(metric_dict.keys())
            
            for metric_name, metric_values in metric_dict.items():
                corr_coef, p_val, _, _ = self.analyze_correlation(
                    clinical_scores, metric_values
                )
                correlations[metric_name] = {
                    'correlation_coefficient': corr_coef,
                    'p_value': p_val
                }
                p_values.append(p_val)
            
            # Apply multiple comparison correction
            p_values = np.array(p_values)
            corrected_p_values = self._apply_multiple_comparison_correction(p_values)
            
            # Update results with corrected p-values
            for i, metric_name in enumerate(metric_names):
                correlations[metric_name]['corrected_p_value'] = float(corrected_p_values[i])
                correlations[metric_name]['is_significant'] = (
                    corrected_p_values[i] < self.significance_alpha
                )
            
            return correlations
            
        except Exception as e:
            if isinstance(e, StatisticalError):
                raise
            raise StatisticalError(f"Multiple correlation analysis failed: {str(e)}")


class RecurrenceDeficitAnalyzer:
    """
    Comprehensive Recurrence Deficit Analysis across Mechanisms.
    
    This class integrates all statistical tests to provide a unified framework
    for analyzing recurrence deficits across the three disentangled mechanisms.
    """
    
    def __init__(
        self,
        significance_alpha: float = 0.05,
        random_state: int = 42
    ):
        """
        Initialize recurrence deficit analyzer.
        
        Parameters
        ----------
        significance_alpha : float, optional
            Significance level (default: 0.05).
        random_state : int, optional
            Random seed for reproducibility (default: 42).
        """
        self.significance_alpha = float(significance_alpha)
        self.random_state = int(random_state)
        
        # Initialize component testers
        self.modulation_tester = ConditionalIndependenceTester(
            test_method='permutation',
            significance_alpha=significance_alpha,
            random_state=random_state
        )
        
        self.quantization_tester = QuantizationSensitivityTester(
            significance_alpha=significance_alpha
        )
        
        self.clinical_analyzer = ClinicalCorrelationAnalyzer(
            correlation_method='spearman',
            significance_alpha=significance_alpha,
            multiple_comparison_correction='fdr'
        )
    
    def analyze_group_differences(
        self,
        metrics: Dict[str, Dict[str, np.ndarray]],
        clinical_groups: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze group differences across all three mechanisms.
        
        Parameters
        ----------
        metrics : dict
            Nested dictionary: metrics[mechanism][group] = values
        clinical_groups : list
            List of clinical group names.
        
        Returns
        -------
        dict
            Comprehensive analysis results.
        """
        try:
            results = {}
            
            # Analyze attractor stability (R_hat)
            if 'attractor_stability' in metrics:
                attractor_groups = [metrics['attractor_stability'][group] for group in clinical_groups]
                h_stat, p_val, is_sig = self.quantization_tester.kruskal_wallis_test(*attractor_groups)
                results['attractor_stability'] = {
                    'test_statistic': h_stat,
                    'p_value': p_val,
                    'is_significant': is_sig,
                    'group_comparisons': {}
                }
            
            # Analyze modulation (Gamma)
            if 'modulation' in metrics:
                modulation_groups = [metrics['modulation'][group] for group in clinical_groups]
                h_stat, p_val, is_sig = self.quantization_tester.kruskal_wallis_test(*modulation_groups)
                results['modulation'] = {
                    'test_statistic': h_stat,
                    'p_value': p_val,
                    'is_significant': is_sig,
                    'group_comparisons': {}
                }
            
            # Analyze quantization sensitivity (Q_tilde)
            if 'quantization_sensitivity' in metrics:
                quantization_groups = [metrics['quantization_sensitivity'][group] for group in clinical_groups]
                h_stat, p_val, is_sig = self.quantization_tester.kruskal_wallis_test(*quantization_groups)
                results['quantization_sensitivity'] = {
                    'test_statistic': h_stat,
                    'p_value': p_val,
                    'is_significant': is_sig,
                    'group_comparisons': {}
                }
            
            # Special AD vs FTD comparison for quantization sensitivity
            if ('quantization_sensitivity' in metrics and 
                'AD' in metrics['quantization_sensitivity'] and 
                'FTD' in metrics['quantization_sensitivity']):
                
                u_stat, p_val, is_sig = self.quantization_tester.mann_whitney_u_test(
                    metrics['quantization_sensitivity']['AD'],
                    metrics['quantization_sensitivity']['FTD']
                )
                results['quantization_sensitivity']['ad_ftd_comparison'] = {
                    'u_statistic': u_stat,
                    'p_value': p_val,
                    'is_significant': is_sig
                }
            
            return results
            
        except Exception as e:
            raise StatisticalError(f"Group differences analysis failed: {str(e)}")
    
    def analyze_clinical_correlations(
        self,
        clinical_scores: np.ndarray,
        attractor_stability: np.ndarray,
        modulation: np.ndarray,
        quantization_sensitivity: np.ndarray,
        mci_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze clinical correlations for all three mechanisms.
        
        Parameters
        ----------
        clinical_scores : np.ndarray
            Clinical scores (MMSE).
        attractor_stability : np.ndarray
            Attractor stability metrics.
        modulation : np.ndarray
            Modulation metrics.
        quantization_sensitivity : np.ndarray
            Quantization sensitivity metrics.
        mci_mask : np.ndarray, optional
            Boolean mask for MCI subgroup.
        
        Returns
        -------
        dict
            Clinical correlation analysis results.
        """
        try:
            metric_dict = {
                'attractor_stability': attractor_stability,
                'quantization_sensitivity': quantization_sensitivity
            }
            
            # Full cohort correlations
            full_correlations = self.clinical_analyzer.analyze_multiple_correlations(
                clinical_scores, metric_dict
            )
            
            results = {
                'full_cohort': full_correlations,
                'mci_subgroup': {}
            }
            
            # MCI subgroup correlation for modulation
            if mci_mask is not None:
                corr_coef, p_val, is_sig, corrected_p = self.clinical_analyzer.analyze_correlation(
                    clinical_scores[mci_mask],
                    modulation[mci_mask]
                )
                results['mci_subgroup']['modulation'] = {
                    'correlation_coefficient': corr_coef,
                    'p_value': p_val,
                    'corrected_p_value': corrected_p,
                    'is_significant': is_sig
                }
            
            return results
            
        except Exception as e:
            raise StatisticalError(f"Clinical correlation analysis failed: {str(e)}")