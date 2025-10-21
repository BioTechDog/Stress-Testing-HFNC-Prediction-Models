import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import interp1d
from scipy.stats import laplace
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


RR_base = np.array([
    9.87952, 12.8916, 13.2874, 13.8038, 15.1119, 15.4045, 15.6799, 16.3855, 
    16.8847, 17.3838, 17.5904, 19.1738, 19.673, 20.7573, 20.9811, 21.2909, 
    21.5835, 22.4957, 22.9776, 23.58, 24.0792, 25.4733, 25.8864, 30.8778, 32.9776
])

RR_mean = np.array([
    9.64286, 11.6667, 12.619, 13.8393, 15.1488, 15.119, 16.0119, 17.1726, 
    16.6369, 17.2321, 18.125, 19.9405, 20.6845, 20.7143, 21.131, 21.5774, 
    21.131, 22.3512, 23.3631, 23.6607, 24.6726, 25.625, 26.9048, 30.2976, 31.6964
])

upper = np.array([
    10.4464, 14.0476, 14.8512, 14.9405, 16.8452, 17.1726, 17.619, 18.3333, 
    17.5595, 17.8571, 19.8214, 21.6964, 23.0655, 22.2619, 22.7381, 23.1845, 
    26.25, 25.3274, 26.1607, 27.5298, 26.5476, 27.2917, 28.6607, 33.006, 33.8988
])

lower = np.array([
    8.72024, 9.73214, 12.1429, 12.5893, 14.3155, 13.7798, 14.2857, 16.369, 
    16.1607, 16.6071, 14.6429, 16.6667, 15.9524, 19.494, 19.7917, 19.9405, 
    17.8571, 20.9226, 20.625, 20.8333, 20.9226, 22.9167, 24.8214, 28.1845, 30.7143
])


HR_MEAN_ERROR = 1.77  # Mean unsigned error (bpm)
HR_PERCENTILES = {
    'p10': 4.0,  
    'p5': 4.0,    
    'p1': 6.0    
}

class RRSimulator:

    
    def __init__(self, RR_base, RR_mean, upper, lower, cv_folds=5):

        self.RR_base = RR_base
        self.RR_mean = RR_mean
        self.upper = upper
        self.lower = lower
        self.cv_folds = cv_folds
        
        self.IQR = upper - lower
        self.sigma = self.IQR / 1.349  

        self.interp_rr_mean = interp1d(
            RR_base, RR_mean, kind='linear', fill_value='extrapolate'
        )
        self.interp_sigma = interp1d(
            RR_base, self.sigma, kind='linear', fill_value='extrapolate'
        )
        self.interp_lower = interp1d(
            RR_base, lower, kind='linear', fill_value='extrapolate'
        )
        self.interp_upper = interp1d(
            RR_base, upper, kind='linear', fill_value='extrapolate'
        )

        self.kde_models = {}
        self._fit_kdes()
    
    def _fit_kdes(self):

        print("Fitting KDE models with cross-validated bandwidth selection...")
        
        for i, rr_base in enumerate(self.RR_base):
            # Get observed statistics
            mu = self.RR_mean[i]
            std = max(self.sigma[i], 1.0)  # Ensure minimum std
            
            training_data = np.random.normal(mu, std, size=500)

            bandwidths = np.linspace(0.1, 5.0, 20)
            grid = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                {'bandwidth': bandwidths},
                cv=self.cv_folds,
                n_jobs=-1
            )
            grid.fit(training_data.reshape(-1, 1))
            
            # Store best KDE model
            best_kde = grid.best_estimator_
            self.kde_models[rr_base] = {
                'kde': best_kde,
                'bandwidth': grid.best_params_['bandwidth'],
                'mu': mu,
                'std': std
            }
            
            print(f"  RR={rr_base:.2f}: bandwidth={grid.best_params_['bandwidth']:.3f}")
    
    def generate_samples(self, RR_true, n_samples=100, clip=True):
        
        RR_true = np.atleast_1d(RR_true)
        all_samples = []
        
        for rr in RR_true:
            mu = float(self.interp_rr_mean(rr))
            std = float(max(self.interp_sigma(rr), 1.0))
            lower_bound = float(self.interp_lower(rr))
            upper_bound = float(self.interp_upper(rr))
            
            if rr in self.kde_models:
                kde_model = self.kde_models[rr]['kde']
            else:
                training_data = np.random.normal(mu, std, size=500)
                
                bandwidths = np.linspace(0.1, 5.0, 20)
                grid = GridSearchCV(
                    KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=self.cv_folds,
                    n_jobs=-1
                )
                grid.fit(training_data.reshape(-1, 1))
                kde_model = grid.best_estimator_
            
            samples = kde_model.sample(n_samples).flatten()
            
            if clip:
                samples = np.clip(samples, lower_bound, upper_bound)
            
            all_samples.append(samples)
        
        return np.array(all_samples).squeeze()
    
    def validate_fit(self, n_bins=5, n_simulations=1000):
       
        print("\nPerforming simulation-based predictive checks...")
        
        bin_edges = np.percentile(self.RR_base, np.linspace(0, 100, n_bins + 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        results = []
        
        for i in range(n_bins):
            mask = (self.RR_base >= bin_edges[i]) & (self.RR_base < bin_edges[i + 1])
            rr_in_bin = self.RR_base[mask]
            
            if len(rr_in_bin) == 0:
                continue

            observed_errors = []
            for rr_base, rr_mean in zip(self.RR_base[mask], self.RR_mean[mask]):
                observed_errors.append(rr_mean - rr_base)

            simulated_errors = []
            for rr_true in rr_in_bin:
                samples = self.generate_samples(rr_true, n_samples=n_simulations)
                errors = samples - rr_true
                simulated_errors.extend(errors)

            results.append({
                'Bin_Center': bin_centers[i],
                'Observed_Median_Error': np.median(observed_errors),
                'Simulated_Median_Error': np.median(simulated_errors),
                'Observed_IQR': np.percentile(observed_errors, 75) - np.percentile(observed_errors, 25),
                'Simulated_IQR': np.percentile(simulated_errors, 75) - np.percentile(simulated_errors, 25)
            })
        
        return pd.DataFrame(results)


class HRSimulator:
   
    
    def __init__(self, mean_error, percentiles):
    
        self.mean_error = mean_error
        self.percentiles = percentiles
        
        # Fit GLD 
        self.gld_params = self._fit_gld()
    
    def _fit_gld(self):

        print("\nFitting Generalized Lambda Distribution for HR errors...")
        
        target_p95 = self.percentiles['p5']  
        target_p99 = self.percentiles['p1']  
        
        scale = target_p95 / (-np.log(0.05))
        
        location = self.mean_error * 0.3 
        
        params = {
            'location': location,
            'scale': scale,
            'distribution': 'laplace'
        }
        
        print(f"  GLD Parameters:")
        print(f"    Location (μ): {location:.3f} bpm")
        print(f"    Scale (b): {scale:.3f} bpm")
        
        return params
    
    def generate_samples(self, true_hr, n_samples=1, clip=True, 
                        min_hr=40, max_hr=200):
      
        true_hr = np.atleast_1d(true_hr)
        
        errors = np.random.laplace(
            loc=self.gld_params['location'],
            scale=self.gld_params['scale'],
            size=(len(true_hr), n_samples)
        )
        
        measured_hr = true_hr.reshape(-1, 1) + errors
        
        if clip:
            measured_hr = np.clip(measured_hr, min_hr, max_hr)
        
        return measured_hr.squeeze()
    
