/**
 * Benchmark Runner for JavaScript PV Models
 * 
 * Runs both physics_model.js and surrogate_model.js on the same input data
 * and outputs predictions with timing information.
 * 
 * Usage: node benchmark_js.js
 * Output: ../output/js_benchmark_results.json
 */

const fs = require('fs');
const path = require('path');

// =====================================================================
// IMPORT PHYSICS MODEL FUNCTIONS (inline since it doesn't export)
// =====================================================================

// We need to extract the calcPV function from physics_model.js
// Since it's designed for ThingsBoard, we'll re-implement the core logic here

const PHYSICS_CONFIG = {
    lat: 8.342368984714714,
    lon: 80.37623529556957,
    tz_offset_h: 5.5,
    orientations: [
        {tilt: 18, az: 148, mods: 18},
        {tilt: 18, az: -32, mods: 18},
        {tilt: 19, az: 55, mods: 36},
        {tilt: 19, az: -125, mods: 36},
        {tilt: 18, az: -125, mods: 36},
        {tilt: 18, az: 55, mods: 36},
        {tilt: 27, az: -125, mods: 18},
        {tilt: 27, az: 55, mods: 18}
    ],
    mod_area: 2.556,
    mod_eff: 0.2153,
    temp_coeff: -0.00340,
    inv_rating_kw: 55,
    inv_eff: 0.98,
    inv_threshold_kw: 0.0,
    far_shade: 1.0,
    albedo: 0.20,
    dc_losses: 0.9317,
    iam_ang: [0, 25, 45, 60, 65, 70, 75, 80, 90],
    iam_val: [1.000, 1.000, 0.995, 0.962, 0.936, 0.903, 0.851, 0.754, 0.000],
    sapm_a: -3.56,
    sapm_b: -0.075,
    sapm_dt: 3,
    total_area: 552.096,
    rated_stc_kw: 118.8
};

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

function clip(v, min, max) {
    return v < min ? min : (v > max ? max : v);
}

function lerp(x, x0, x1, y0, y1) {
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

function interpIAM(aoi) {
    const ang = PHYSICS_CONFIG.iam_ang;
    const val = PHYSICS_CONFIG.iam_val;
    if (aoi <= ang[0]) return val[0];
    if (aoi >= ang[8]) return val[8];
    
    for (let i = 0; i < 8; i++) {
        if (aoi <= ang[i + 1]) {
            return lerp(aoi, ang[i], ang[i + 1], val[i], val[i + 1]);
        }
    }
    return 0;
}

function solarPos(ts, lat, lon) {
    const d = new Date(ts);
    const jd = d.getTime() / 86400000 + 2440587.5;
    const jc = (jd - 2451545) / 36525;
    
    const m = (357.52911 + jc * 35999.05029) * DEG2RAD;
    const c = (1.914602 - jc * 0.004817) * Math.sin(m) + 
            0.019993 * Math.sin(2 * m) + 
            0.000289 * Math.sin(3 * m);
    const sunLon = (280.46646 + jc * 36000.76983 + c) * DEG2RAD;
    
    const obl = (23.439291 - jc * 0.0130042) * DEG2RAD;
    const dec = Math.asin(Math.sin(obl) * Math.sin(sunLon));
    
    const eot = 4 * (sunLon * RAD2DEG - Math.atan2(
        Math.cos(obl) * Math.sin(sunLon),
        Math.cos(sunLon)
    ) * RAD2DEG);
    
    const utcH = d.getUTCHours() + d.getUTCMinutes() / 60;
    const solarT = utcH + lon / 15 + eot / 60;
    const ha = (solarT - 12) * 15 * DEG2RAD;
    
    const latR = lat * DEG2RAD;
    const sinEl = Math.sin(latR) * Math.sin(dec) + 
                Math.cos(latR) * Math.cos(dec) * Math.cos(ha);
    const el = Math.asin(clip(sinEl, -1, 1));
    
    const cosAz = (Math.sin(dec) - Math.sin(latR) * sinEl) / 
                (Math.cos(latR) * Math.cos(el));
    let az = Math.acos(clip(cosAz, -1, 1));
    if (ha > 0) az = 2 * Math.PI - az;
    
    return {
        zen: 90 - el * RAD2DEG,
        az: az * RAD2DEG,
        el: el * RAD2DEG
    };
}

function perezPOA(ghi, dni, dhi, sun, tilt, azim, alb) {
    const zenR = sun.zen * DEG2RAD;
    const tiltR = tilt * DEG2RAD;
    const azR = azim * DEG2RAD;
    const sunAzR = sun.az * DEG2RAD;
    
    let cosAOI = Math.cos(zenR) * Math.cos(tiltR) +
                 Math.sin(zenR) * Math.sin(tiltR) * Math.cos(sunAzR - azR);
    cosAOI = clip(cosAOI, -1, 1);
    const aoi = Math.acos(cosAOI) * RAD2DEG;
    
    const beam = (cosAOI > 0 && sun.el > 0) ? dni * cosAOI : 0;
    
    const f = 0.5 + 0.5 * Math.cos(tiltR);
    let diff = dhi * f;
    
    if (dni > 50 && cosAOI > 0.087) {
        const circum = dhi * 0.2 * (cosAOI / Math.max(0.087, Math.cos(zenR)));
        diff += circum;
    }
    
    const ground = ghi * alb * (1 - Math.cos(tiltR)) * 0.5;
    
    return {
        poa: Math.max(0, beam + diff + ground),
        aoi: aoi
    };
}

function calcPVPhysics(data) {
    const ghi = Math.max(0, data.ghi || 0);
    const dni = Math.max(0, data.dni || 0);
    const dhi = Math.max(0, data.dhi || 0);
    const tAmb = data.air_temp || 25;
    const wind = data.wind_speed || 1.0;
    
    const sun = solarPos(data.timestamp, PHYSICS_CONFIG.lat, PHYSICS_CONFIG.lon);
    if (sun.el < 0 || ghi < 1) {
        return { ac: 0, dc: 0, tcell: tAmb, pr: 0 };
    }
    
    let totalDC = 0;
    let totalTCell = 0;
    const orientCount = PHYSICS_CONFIG.orientations.length;
    
    for (let i = 0; i < orientCount; i++) {
        const o = PHYSICS_CONFIG.orientations[i];
        const areaFrac = (o.mods * PHYSICS_CONFIG.mod_area) / PHYSICS_CONFIG.total_area;
        
        const poa = perezPOA(ghi, dni, dhi, sun, o.tilt, o.az, PHYSICS_CONFIG.albedo);
        const poaShaded = poa.poa * PHYSICS_CONFIG.far_shade;
        
        const iam = interpIAM(poa.aoi);
        const poaOpt = poaShaded * iam;
        
        const e0 = poaShaded / 1000;
        const tCell = tAmb + PHYSICS_CONFIG.sapm_a * e0 + PHYSICS_CONFIG.sapm_b * e0 * wind + PHYSICS_CONFIG.sapm_dt;
        totalTCell += tCell;
        
        let dcKwM2 = poaOpt * PHYSICS_CONFIG.mod_eff / 1000;
        dcKwM2 *= (1 + PHYSICS_CONFIG.temp_coeff * (tCell - 25));
        dcKwM2 *= PHYSICS_CONFIG.dc_losses;
        
        totalDC += dcKwM2 * PHYSICS_CONFIG.total_area * areaFrac;
    }
    
    if (totalDC < PHYSICS_CONFIG.inv_threshold_kw) totalDC = 0;
    
    const ac = Math.min(totalDC * PHYSICS_CONFIG.inv_eff, PHYSICS_CONFIG.inv_rating_kw);
    
    return { ac: ac, dc: totalDC, tcell: totalTCell / orientCount };
}

// =====================================================================
// IMPORT SURROGATE MODEL
// =====================================================================

// Path: shared/ -> ../js_surrogate/
const surrogateModel = require('../js_surrogate/surrogate_model.js');

// =====================================================================
// CSV PARSING
// =====================================================================

function parseCSV(filepath) {
    const content = fs.readFileSync(filepath, 'utf8');
    const lines = content.trim().split('\n');
    const headers = lines[0].split(',');
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        headers.forEach((h, idx) => {
            row[h.trim()] = values[idx] ? values[idx].trim() : '';
        });
        data.push(row);
    }
    return data;
}

// =====================================================================
// MAIN BENCHMARK
// =====================================================================

function runBenchmark() {
    console.log('='.repeat(60));
    console.log('JAVASCRIPT MODEL BENCHMARK');
    console.log('='.repeat(60));
    
    // Load input data (scripts/shared/ -> ../../data/)
    const inputPath = path.join(__dirname, '..', '..', 'data', 'solcast_irradiance.csv');
    console.log(`\nLoading: ${inputPath}`);
    
    const rawData = parseCSV(inputPath);
    console.log(`Loaded ${rawData.length} hourly records`);
    
    // Prepare data with proper column mapping
    const data = rawData.map(row => ({
        timestamp: row.period_end,
        ghi: parseFloat(row.ghi) || 0,
        dni: parseFloat(row.dni) || 0,
        dhi: parseFloat(row.dhi) || 0,
        air_temp: parseFloat(row.air_temp) || 25,
        wind_speed: parseFloat(row.wind_speed_10m) || 1.0
    }));
    
    // Results storage
    const results = {
        physics_js: [],
        surrogate_js: [],
        timing: {
            physics_js: { total_ms: 0, predictions: 0 },
            surrogate_js: { total_ms: 0, predictions: 0 }
        }
    };
    
    // =====================================================================
    // RUN PHYSICS MODEL (JS)
    // =====================================================================
    console.log('\n--- Running JS Physics Model ---');
    
    const physicsStart = process.hrtime.bigint();
    
    for (const row of data) {
        const t0 = process.hrtime.bigint();
        const result = calcPVPhysics(row);
        const t1 = process.hrtime.bigint();
        
        results.physics_js.push({
            timestamp: row.timestamp,
            ac_power_kw: result.ac,
            time_ns: Number(t1 - t0)
        });
        
        results.timing.physics_js.predictions++;
    }
    
    const physicsEnd = process.hrtime.bigint();
    results.timing.physics_js.total_ms = Number(physicsEnd - physicsStart) / 1e6;
    
    console.log(`  Predictions: ${results.timing.physics_js.predictions}`);
    console.log(`  Total time: ${results.timing.physics_js.total_ms.toFixed(2)} ms`);
    console.log(`  Avg per prediction: ${(results.timing.physics_js.total_ms / results.timing.physics_js.predictions).toFixed(3)} ms`);
    
    // =====================================================================
    // RUN SURROGATE MODEL (JS)
    // =====================================================================
    console.log('\n--- Running JS Surrogate Model ---');
    
    const surrogateStart = process.hrtime.bigint();
    
    for (const row of data) {
        const t0 = process.hrtime.bigint();
        const result = surrogateModel.predictPV({
            ghi: row.ghi,
            dni: row.dni,
            dhi: row.dhi,
            airTemp: row.air_temp,
            windSpeed: row.wind_speed
        });
        const t1 = process.hrtime.bigint();
        
        results.surrogate_js.push({
            timestamp: row.timestamp,
            ac_power_kw: result.powerKW,
            time_ns: Number(t1 - t0)
        });
        
        results.timing.surrogate_js.predictions++;
    }
    
    const surrogateEnd = process.hrtime.bigint();
    results.timing.surrogate_js.total_ms = Number(surrogateEnd - surrogateStart) / 1e6;
    
    console.log(`  Predictions: ${results.timing.surrogate_js.predictions}`);
    console.log(`  Total time: ${results.timing.surrogate_js.total_ms.toFixed(2)} ms`);
    console.log(`  Avg per prediction: ${(results.timing.surrogate_js.total_ms / results.timing.surrogate_js.predictions).toFixed(3)} ms`);
    
    // =====================================================================
    // SAVE RESULTS
    // =====================================================================
    // Path: scripts/shared/ -> ../../output/
    const outputPath = path.join(__dirname, '..', '..', 'output', 'js_benchmark_results.json');
    
    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    
    fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
    console.log(`\nResults saved to: ${outputPath}`);
    
    console.log('\n' + '='.repeat(60));
    console.log('BENCHMARK COMPLETE');
    console.log('='.repeat(60));
    
    return results;
}

// Run if executed directly
if (require.main === module) {
    runBenchmark();
}

module.exports = { runBenchmark };

