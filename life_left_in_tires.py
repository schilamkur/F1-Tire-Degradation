import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns    
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution
import warnings

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache('f1_cache')
plotting.setup_mpl()

BAD_STATUSES = {
    'Accident', 'Collision', 'Retired', 'Damage',
    'DNF', 'Did not finish', 'Disqualified'
}

def extract_stint_degradation(session):
    session.load()
    laps = session.laps.pick_accurate()
    results = session.results[['DriverNumber', 'Status']]
    laps = laps.merge(results, how='left', on='DriverNumber')
    weather = session.weather_data

    data = []
    drivers = laps['Driver'].unique()
    for driver in tqdm(drivers, desc="Driver processing"):
        driver_laps = laps.pick_drivers([driver])
        stints = driver_laps['Stint'].unique()

        for stint_num in stints:
            stint_laps = driver_laps[driver_laps['Stint'] == stint_num].sort_values('LapNumber')
            if len(stint_laps) < 7:
                continue
            if any(stint_laps['Status'].isin(BAD_STATUSES)):
                continue
            lap_times = stint_laps['LapTime'].dt.total_seconds()
            if lap_times.isna().any():
                continue

            degradation_lap_index = None
            for i in range(3, len(lap_times)):
                rolling_mean = lap_times.iloc[:i].rolling(window=3).mean().iloc[-1]
                rolling_std = lap_times.iloc[:i].rolling(window=3).std().iloc[-1]
                if rolling_std is np.nan:
                    continue
                threshold = rolling_mean + 1.5 * rolling_std
                if lap_times.iloc[i] > threshold:
                    degradation_lap_index = i
                    break

            if degradation_lap_index is None:
                continue

            degradation_lap_number = stint_laps.iloc[degradation_lap_index]['LapNumber']
            stint_start_lap = stint_laps.iloc[0]['LapNumber']
            tire_life = degradation_lap_number - stint_start_lap
            starting_position = stint_laps.iloc[0]['Position']

            total_laps = laps['LapNumber'].max()
            laps_remaining_when_stint_started = total_laps - stint_start_lap

            avg_lap_time_first_3 = lap_times.iloc[:3].mean()
            avg_lap_time_last_3 = lap_times.iloc[-3:].mean()
            lap_time_diff_first_last = avg_lap_time_last_3 - avg_lap_time_first_3

            tire_age_at_start = stint_laps.iloc[0]['TyreLife']
            track_status_pct = (stint_laps['TrackStatus'] != 1).sum() / len(stint_laps)

            start_lap = stint_laps.iloc[0]['LapNumber']
            pit_laps = laps[laps['PitOutTime'].notna()]
            nearby_pit_count = pit_laps[
                pit_laps['LapNumber'].between(start_lap - 2, start_lap + 2)
            ].shape[0]

            driver_all_laps = laps[laps['Driver'] == driver]
            prior_stint_laps = driver_all_laps[
                driver_all_laps['LapNumber'] < start_lap
            ]
            prior_pit_count = prior_stint_laps['PitInTime'].notna().sum()

            compound = stint_laps.iloc[0]['Compound']
            fresh = stint_laps.iloc[0]['FreshTyre']
            team = stint_laps.iloc[0]['Team']
            track = session.event['EventName']
            location = session.event['Location']
            date = session.event['EventDate']

            data.append({
                'driver': driver,
                'team': team,
                'track': track,
                'location': location,
                'date': date,
                'compound': compound,
                'fresh_tyre': fresh,
                'stint_num': stint_num,
                'laps_in_stint': len(stint_laps),
                'tire_life': tire_life,
                'avg_lap_time_first_3': avg_lap_time_first_3,
                'avg_lap_time_last_3': avg_lap_time_last_3,
                'lap_time_diff_first_last': lap_time_diff_first_last,
                'temp_C': weather['AirTemp'].mean(),
                'humidity_%': weather['Humidity'].mean(),
                'pressure_hPa': weather['Pressure'].mean(),
                'rainfall_mm': weather['Rainfall'].mean(),
                'wind_speed_kmh': weather['WindSpeed'].mean(),
                'wind_direction_deg': weather['WindDirection'].mean(),
                'starting_position': starting_position,
                'laps_remaining_when_stint_started': laps_remaining_when_stint_started,
                'tire_age_at_start': tire_age_at_start,
                'track_status_flag_rate': track_status_pct,
                'nearby_pit_count': nearby_pit_count,
                'prior_pit_count': prior_pit_count,
            })

    return pd.DataFrame(data)


def extract_data_over_years(years):
    total_data = []
    for year in years:
        print(f"Processing season {year}")
        schedule = fastf1.get_event_schedule(year)
        for _, event in schedule.iterrows():
            event_name = event['EventName']
            try:
                session = fastf1.get_session(year, event_name, 'R')
                df = extract_stint_degradation(session)
                total_data.append(df)
            except Exception as e:
                print(f"Failed to process {event_name}: {e}")

    return pd.concat(total_data, ignore_index=True)


def plot_stint_with_degradation(session, driver, stint_num):
    session.load()
    laps = session.laps.pick_accurate()
    driver_laps = laps.pick_drivers([driver])
    stint_laps = driver_laps[driver_laps['Stint'] == stint_num].sort_values('LapNumber')

    lap_times = stint_laps['LapTime'].dt.total_seconds()
    degradation_lap = None

    for i in range(3, len(lap_times)):
        avg_so_far = lap_times.iloc[:i].mean()
        if lap_times.iloc[i] > avg_so_far + 1.0:
            degradation_lap = stint_laps.iloc[i]['LapNumber']
            break

    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stint_laps['LapNumber'], lap_times, marker='o', label='Lap Time')

    if degradation_lap:
        ax.axvline(degradation_lap, color='red', linestyle='--', label='Degradation Lap')

    ax.set_title(f"{session.event['EventName']} - {driver} - Stint {stint_num}")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # get data
    '''
    os.makedirs('f1_cache', exist_ok=True)
    fastf1.Cache.enable_cache('f1_cache')

    years = [2022, 2023, 2024, 2025]
    full_df = extract_data_over_years(years)
    full_df.to_csv('tire_degradation.csv', index=False)
    '''
    # plotting
    '''
    session = fastf1.get_session(2023, 'Monza', 'R')
    plot_stint_with_degradation(session, driver='HAM', stint_num=1)
    '''

    # running models
    df = pd.read_csv('tire_degradation.csv').dropna()
    df = df[df['tire_life'].between(df['tire_life'].quantile(0.05), df['tire_life'].quantile(0.95))]

    df['driver_avg_tire_life'] = df.groupby('driver')['tire_life'].transform('mean')
    df['team_avg_tire_life'] = df.groupby('team')['tire_life'].transform('mean')

    track_type_map = {
        'Monza': 'power',
        'Spa-Francorchamps': 'power',
        'Silverstone': 'balanced',
        'Suzuka': 'technical',
        'Hungaroring': 'technical',
        'Singapore': 'street',
        'Monaco': 'street',
        'Baku': 'street',
        'Miami': 'street',
        'Zandvoort': 'technical',
        'Imola': 'technical',
        'Barcelona': 'balanced',
        'Austria': 'balanced',
        'Bahrain': 'power',
        'Jeddah': 'street',
        'Las Vegas': 'street',
        'Qatar': 'power',
        'COTA': 'balanced',
        'Mexico City': 'power',
        'Brazil': 'balanced',
        'Abu Dhabi': 'balanced',
    }

    df['track_type'] = df['track'].map(track_type_map).fillna('balanced')

    engine_map = {
        'Red Bull Racing': 'Honda',
        'AlphaTauri': 'Honda',
        'Aston Martin': 'Mercedes',
        'Mercedes': 'Mercedes',
        'McLaren': 'Mercedes',
        'Ferrari': 'Ferrari',
        'Haas F1 Team': 'Ferrari',
        'Alpine': 'Renault',
        'Williams': 'Mercedes',
        'Alfa Romeo': 'Ferrari'
    }
    df['engine_manufacturer'] = df['team'].map(engine_map).fillna('Unknown')

    categorical_features = ['compound', 'fresh_tyre', 'driver', 'team', 'track', 'track_type', 
                            'engine_manufacturer']
    numeric_features = [
        'avg_lap_time_first_3', 'avg_lap_time_last_3', 'lap_time_diff_first_last',
        'temp_C', 'humidity_%', 'pressure_hPa', 'rainfall_mm', 'wind_speed_kmh', 
        'wind_direction_deg', 'driver_avg_tire_life', 'team_avg_tire_life', 
        'starting_position', 'laps_remaining_when_stint_started', 'tire_age_at_start', 
        'track_status_flag_rate', 'nearby_pit_count', 'prior_pit_count'
    ]

    X = df[categorical_features + numeric_features]
    y = df['tire_life']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pt = PowerTransformer(method='yeo-johnson')
    y_train_trans = pt.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_trans = pt.transform(y_test.values.reshape(-1, 1)).ravel()

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoost': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
    }

    param_spaces = {
        'RandomForest': {
            'regressor__n_estimators': IntDistribution(50, 300),
            'regressor__max_depth': IntDistribution(5, 30),
        },
        'GradientBoost': {
            'regressor__n_estimators': IntDistribution(100, 300),
            'regressor__learning_rate': FloatDistribution(0.01, 0.2),
            'regressor__max_depth': IntDistribution(3, 10)
        },
        'XGBoost': {
            'regressor__n_estimators': IntDistribution(100, 300),
            'regressor__learning_rate': FloatDistribution(0.01, 0.2),
            'regressor__max_depth': IntDistribution(3, 10)
        },
        'LightGBM': {
            'regressor__n_estimators': IntDistribution(100, 300),
            'regressor__learning_rate': FloatDistribution(0.01, 0.2),
            'regressor__max_depth': IntDistribution(3, 10)
        },
        'CatBoost': {
            'regressor__iterations': IntDistribution(100, 300),
            'regressor__learning_rate': FloatDistribution(0.01, 0.2),
            'regressor__depth': IntDistribution(4, 10)
        }
    }

    results = []

    for name, model in models.items():
        print(f"\nTuning {name} with Optuna...")

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        search = OptunaSearchCV(
            pipe,
            param_distributions=param_spaces[name],
            cv=5,
            n_trials=50,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train_trans)
        y_pred_trans = search.predict(X_test)
        y_pred = pt.inverse_transform(y_pred_trans.reshape(-1, 1)).ravel()

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Best params for {name}: {search.best_params_}")
        print(f"MAE: {mae:.3f}, R2: {r2:.3f}")

        results.append({
            'Model': name,
            'Best Params': search.best_params_,
            'MAE': mae,
            'R2': r2
        })

    results_df = pd.DataFrame(results)
    results_df['MAE'] = results_df['MAE'].round(3)
    results_df['R2'] = results_df['R2'].round(6)
    results_df = results_df.sort_values(by='MAE')
    print("Final Results (sorted by MAE):")
    print(results_df.to_string(index=False))

    # determine most important features
    best_model_name = results_df.iloc[0]['Model']
    print(f"\nRunning SHAP analysis on: {best_model_name}")
    best_model = models[best_model_name]
    best_params = results_df.iloc[0]['Best Params']

    flat_params = {k.replace('regressor__', ''): v for k, v in best_params.items()}
    best_model.set_params(**flat_params)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', best_model)
    ])

    pipeline.fit(X_train, y_train)

    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

    regressor_model = pipeline.named_steps['regressor']
    if hasattr(regressor_model, 'get_booster') or hasattr(regressor_model, 'feature_importances_'):
        explainer = shap.Explainer(regressor_model)
    else:
        explainer = shap.Explainer(regressor_model.predict, X_test_transformed)

    shap_values = explainer(X_test_transformed)

    feature_names_cat = pipeline.named_steps['preprocessor'].named_transformers_['cat'] \
        .get_feature_names_out(categorical_features)
    feature_names_total = list(feature_names_cat) + numeric_features

    X_test_named = pd.DataFrame(X_test_transformed.toarray() if hasattr(X_test_transformed, 'toarray') else X_test_transformed,
                                columns=feature_names_total)

    print("\nGenerating SHAP summary plot...")
    shap.plots.beeswarm(shap_values, max_display=25)

    print("\nGenerating SHAP bar plot...")
    shap.plots.bar(shap_values, max_display=25)
        