/**
 * PV Generation Surrogate Model - JavaScript Implementation
 * 
 * PURPOSE:
 * Approximates the output of a physics-based Python PV model using
 * regression-based surrogate modeling. Designed for ThingsBoard Rule Chains.
 * 
 * IMPORTANT NOTES:
 * 1. This is NOT a physics engine - it's a trained approximation
 * 2. Coefficients MUST be fitted to your specific plant's data
 * 3. Valid only within training data bounds (irradiance, temperature)
 * 4. For testing and benchmarking only
 * 
 * COEFFICIENTS (PLACEHOLDER - REPLACE AFTER TRAINING):
 * These values are ILLUSTRATIVE and will NOT work for your plant.
 * You must fit them using your CSV data.
 */

const REGRESSION_COEFFICIENTS = {
    // FITTED REGRESSION COEFFICIENTS FOR 55kW PLANT
    // Generated from 144 hourly samples
    // Validation: R²=0.9901, MAE=0.77 kW, Energy Error=0.64%
    
    a0: -0.281678,      // Intercept
    a1: 0.099291,       // Linear GHI term
    a2: 5.682749e-06,   // Quadratic GHI term
    a3: -0.010893,      // DNI contribution
    a4: -2.031135e-03,  // Temperature interaction (GHI × ΔT)
    a5: -6.875986e-04   // Wind interaction (GHI × wind_speed)
  };
  
  /**
   * Plant parameters (static, from PVsyst report)
   * These are passed in but could be hardcoded for a specific deployment
   */
  const PLANT_PARAMS = {
    // Total module area (m²) - sum of all orientations
    totalModuleArea: 216 * 2.556,  // 216 modules × 2.556 m²/module
    
    // Module efficiency at STC (fraction)
    moduleEfficiencySTC: 0.2153,
    
    // Power temperature coefficient (per °C)
    gammaP: -0.00340,
    
    // Combined DC loss factor (accounts for soiling, LID, module quality, mismatch, wiring)
    // From config: (1-0.03)×(1-0.014)×(1+0.008)×(1-0.017)×(1-0.009) ≈ 0.9317
    dcLossFactor: 0.9317,
    
    // AC wiring loss factor (applied after inverter)
    acLossFactor: 0.997,  // 1 - 0.003
    
    // Inverter AC rating (kW)
    inverterRatingKW: 55.0,
    
    // STC reference temperature (°C)
    tempSTC: 25.0
  };
  
  /**
   * Input validation and bounds checking
   * Prevents nonsensical predictions outside training domain
   */
  function validateInputs(inputs) {
    const bounds = {
      ghi: { min: 0, max: 1400 },      // W/m²
      dni: { min: 0, max: 1200 },      // W/m²
      dhi: { min: 0, max: 800 },       // W/m²
      airTemp: { min: -10, max: 50 },  // °C
      windSpeed: { min: 0, max: 20 }   // m/s
    };
    
    const warnings = [];
    
    for (const [key, value] of Object.entries(inputs)) {
      if (bounds[key]) {
        if (value < bounds[key].min || value > bounds[key].max) {
          warnings.push(`${key}=${value} outside training bounds [${bounds[key].min}, ${bounds[key].max}]`);
        }
      }
    }
    
    return warnings;
  }
  
  /**
   * Core prediction function
   * 
   * @param {Object} inputs - Weather inputs
   * @param {number} inputs.ghi - Global Horizontal Irradiance (W/m²)
   * @param {number} inputs.dni - Direct Normal Irradiance (W/m²)
   * @param {number} inputs.dhi - Diffuse Horizontal Irradiance (W/m²)
   * @param {number} inputs.airTemp - Air temperature (°C)
   * @param {number} inputs.windSpeed - Wind speed at 10m (m/s)
   * 
   * @param {Object} params - Plant parameters (optional, uses defaults if not provided)
   * 
   * @returns {Object} Prediction result
   * @returns {number} result.powerKW - Predicted AC power (kW)
   * @returns {number} result.energyKWh - Energy for timestep (assumes 1 hour)
   * @returns {string[]} result.warnings - Input validation warnings
   */
  function predictPV(inputs, params = PLANT_PARAMS) {
    // Input validation
    const warnings = validateInputs(inputs);
    
    // Destructure inputs
    const { ghi, dni, dhi, airTemp, windSpeed } = inputs;
    const { tempSTC, inverterRatingKW, dcLossFactor, acLossFactor } = params;
    
    // Handle nighttime / zero irradiance
    if (ghi < 1.0) {
      return {
        powerKW: 0.0,
        energyKWh: 0.0,
        warnings: warnings
      };
    }
    
    // Calculate derived features
    const deltaT = airTemp - tempSTC;  // Temperature deviation from STC
    const ghiSquared = ghi * ghi;       // Quadratic term for efficiency curve
    const ghiTempInteraction = ghi * deltaT;  // Temperature loss interaction
    const ghiWindInteraction = ghi * windSpeed;  // Cooling benefit interaction
    
    // Regression equation (core surrogate model)
    const { a0, a1, a2, a3, a4, a5 } = REGRESSION_COEFFICIENTS;
    
    let powerKW = (
      a0 +
      a1 * ghi +
      a2 * ghiSquared +
      a3 * dni +
      a4 * ghiTempInteraction +
      a5 * ghiWindInteraction
    );
    
    // Apply physical constraints
    // 1. Non-negativity (prevent negative predictions at low irradiance)
    powerKW = Math.max(0.0, powerKW);
    
    // 2. Inverter clipping (hard limit at AC rating)
    powerKW = Math.min(powerKW, inverterRatingKW);
    
    // 3. Sanity check: power cannot exceed theoretical DC maximum
    //    P_max ≈ GHI × module_area × efficiency × DC_loss × inverter_eff
    //    Using conservative 0.98 for inverter efficiency
    const theoreticalMaxKW = (ghi / 1000) * params.totalModuleArea * 
                             params.moduleEfficiencySTC * dcLossFactor * 0.98;
    
    if (powerKW > theoreticalMaxKW * 1.1) {  // Allow 10% margin for model noise
      warnings.push(`Prediction ${powerKW.toFixed(1)} kW exceeds physical limit ${theoreticalMaxKW.toFixed(1)} kW`);
      powerKW = theoreticalMaxKW;
    }
    
    // Calculate energy (assuming 1-hour timestep)
    const energyKWh = powerKW * 1.0;
    
    return {
      powerKW: powerKW,
      energyKWh: energyKWh,
      warnings: warnings
    };
  }
  
  /**
   * Batch prediction for hourly timeseries
   * 
   * @param {Array} timeseriesData - Array of {timestamp, ghi, dni, dhi, airTemp, windSpeed}
   * @param {Object} params - Plant parameters (optional)
   * 
   * @returns {Array} Array of prediction results with timestamps
   */
  function predictTimeseries(timeseriesData, params = PLANT_PARAMS) {
    return timeseriesData.map(row => {
      const result = predictPV({
        ghi: row.ghi,
        dni: row.dni,
        dhi: row.dhi,
        airTemp: row.airTemp,
        windSpeed: row.windSpeed
      }, params);
      
      return {
        timestamp: row.timestamp,
        ...result
      };
    });
  }
  
  /**
   * Daily energy aggregation
   * 
   * @param {Array} hourlyPredictions - Output from predictTimeseries()
   * @returns {number} Total daily energy (kWh)
   */
  function aggregateDailyEnergy(hourlyPredictions) {
    return hourlyPredictions.reduce((sum, pred) => sum + pred.energyKWh, 0);
  }
  
  /**
   * ThingsBoard Rule Chain Integration Example
   * 
   * This function can be directly embedded in a ThingsBoard Script node
   * 
   * Expected msg structure:
   * {
   *   "ghi": 850,
   *   "dni": 650,
   *   "dhi": 120,
   *   "airTemp": 28.5,
   *   "windSpeed": 2.3
   * }
   */
  function thingsBoardRuleNode(msg, metadata, msgType) {
    // Extract inputs from ThingsBoard message
    const inputs = {
      ghi: msg.ghi || 0,
      dni: msg.dni || 0,
      dhi: msg.dhi || 0,
      airTemp: msg.airTemp || 25,
      windSpeed: msg.windSpeed || 1.0
    };
    
    // Run prediction
    const result = predictPV(inputs);
    
    // Add predictions to message
    msg.predictedPowerKW = result.powerKW;
    msg.predictedEnergyKWh = result.energyKWh;
    
    // Log warnings if any
    if (result.warnings.length > 0) {
      msg.modelWarnings = result.warnings.join('; ');
    }
    
    // Return modified message
    return {
      msg: msg,
      metadata: metadata,
      msgType: msgType
    };
  }
  
  /**
   * COEFFICIENT FITTING INSTRUCTIONS
   * 
   * To fit coefficients to your CSV data:
   * 
   * 1. Export your CSV to a format with columns:
   *    timestamp, ghi, dni, dhi, air_temp, wind_speed, AC_power_kw
   * 
   * 2. Use Python/R for regression (example in Python):
   * 
   *    import pandas as pd
   *    from sklearn.linear_model import LinearRegression
   *    
   *    # Load data
   *    df = pd.read_csv("your_pv_data.csv")
   *    
   *    # Create feature matrix
   *    df['ghi_squared'] = df['ghi'] ** 2
   *    df['delta_T'] = df['air_temp'] - 25
   *    df['ghi_temp'] = df['ghi'] * df['delta_T']
   *    df['ghi_wind'] = df['ghi'] * df['wind_speed']
   *    
   *    # Define features (X) and target (y)
   *    X = df[['ghi', 'ghi_squared', 'dni', 'ghi_temp', 'ghi_wind']]
   *    y = df['AC_power_kw']
   *    
   *    # Fit regression
   *    model = LinearRegression(fit_intercept=True)
   *    model.fit(X, y)
   *    
   *    # Extract coefficients
   *    print("a0 (intercept):", model.intercept_)
   *    print("a1 (ghi):", model.coef_[0])
   *    print("a2 (ghi²):", model.coef_[1])
   *    print("a3 (dni):", model.coef_[2])
   *    print("a4 (ghi×ΔT):", model.coef_[3])
   *    print("a5 (ghi×wind):", model.coef_[4])
   *    
   *    # Validation
   *    from sklearn.metrics import mean_absolute_error, r2_score
   *    y_pred = model.predict(X)
   *    print("MAE:", mean_absolute_error(y, y_pred), "kW")
   *    print("R²:", r2_score(y, y_pred))
   * 
   * 3. Replace REGRESSION_COEFFICIENTS object above with fitted values
   * 
   * 4. Validate on held-out data (different time period)
   */
  
  // Export functions for module usage (Node.js/ES6)
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      predictPV,
      predictTimeseries,
      aggregateDailyEnergy,
      thingsBoardRuleNode,
      REGRESSION_COEFFICIENTS,
      PLANT_PARAMS
    };
  }
  
  // Example usage (for testing in browser console or Node.js)
  if (typeof window !== 'undefined' || typeof global !== 'undefined') {
    // Example: Single prediction
    const exampleInputs = {
      ghi: 850,        // W/m²
      dni: 650,        // W/m²
      dhi: 120,        // W/m²
      airTemp: 28.5,   // °C
      windSpeed: 2.3   // m/s
    };
    
    const result = predictPV(exampleInputs);
    console.log("Example Prediction:", result);
    
    // Example: Daily timeseries (simulate 24 hours)
    const hourlyData = [];
    for (let hour = 0; hour < 24; hour++) {
      // Simple sinusoidal irradiance pattern (for demonstration)
      const solarAngle = Math.max(0, Math.sin((hour - 6) * Math.PI / 12));
      hourlyData.push({
        timestamp: `2025-12-10T${hour.toString().padStart(2, '0')}:00:00Z`,
        ghi: solarAngle * 900,
        dni: solarAngle * 700,
        dhi: solarAngle * 150,
        airTemp: 25 + 5 * solarAngle,
        windSpeed: 2.0
      });
    }
    
    const predictions = predictTimeseries(hourlyData);
    const dailyEnergy = aggregateDailyEnergy(predictions);
    
    console.log("Daily Energy:", dailyEnergy.toFixed(1), "kWh");
    console.log("Peak Power:", Math.max(...predictions.map(p => p.powerKW)).toFixed(1), "kW");
  }