/**
 * Statik.js - A modern statistics library with optional Chart.js visualization
 */
export class Statik {
    constructor() {
        this.version = '1.0.0';
        this.SQRT_2PI = Math.sqrt(2 * Math.PI);
    }

    // Utility methods
    #validateArray(arr) {
        if (!Array.isArray(arr) || arr.length === 0) {
            throw new Error('Input must be a non-empty array');
        }
        if (!arr.every(x => typeof x === 'number' && !isNaN(x))) {
            throw new Error('All elements must be valid numbers');
        }
    }

    #sortArray(arr) {
        return [...arr].sort((a, b) => a - b);
    }

    #factorial(n) {
        if (n < 0) return NaN;
        if (n === 0) return 1;
        let result = 1;
        for (let i = 2; i <= n; i++) result *= i;
        return result;
    }

    #combination(n, k) {
        return this.#factorial(n) / (this.#factorial(k) * this.#factorial(n - k));
    }

    #beta(a, b) {
        return (this.#factorial(a - 1) * this.#factorial(b - 1)) / this.#factorial(a + b - 1);
    }

    #gamma(z) {
        if (z < 0.5) {
            return Math.PI / (Math.sin(Math.PI * z) * this.#gamma(1 - z));
        }
        z -= 1;
        let x = 0.99999999999980993;
        const p = [
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ];
        for (let i = 0; i < p.length; i++) {
            x += p[i] / (z + i + 1);
        }
        const t = z + p.length - 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
    }

    #logGamma(z) {
        const c = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5
        ];
        
        let sum = c[0];
        const x = z;
        
        for (let i = 1; i < 6; i++) {
            sum += c[i] / (x + i);
        }
        
        const series = sum / x;
        
        return (z + 0.5) * Math.log(z + 5.5) - (z + 5.5) + Math.log(2.5066282746310005 * series / x);
    }

    #gammaIncomplete(x, a) {
        if (x <= 0) return 0;
        if (x > 200) return 1;
        
        const eps = 1e-14;
        let sum = 1 / a;
        let term = sum;
        
        for (let n = 1; n < 100; n++) {
            term *= x / (a + n);
            sum += term;
            if (Math.abs(term) < eps * Math.abs(sum)) break;
        }
        
        return sum * Math.exp(-x + a * Math.log(x) - this.#logGamma(a));
    }

    #betaIncomplete(x, a, b) {
        if (x < 0 || x > 1) return NaN;
        if (x === 0) return 0;
        if (x === 1) return 1;
        
        const bt = Math.exp(
            this.#logGamma(a + b) -
            this.#logGamma(a) -
            this.#logGamma(b) +
            a * Math.log(x) +
            b * Math.log(1 - x)
        );
        
        if (x < (a + 1) / (a + b + 2)) {
            return bt * this.#betaCF(x, a, b) / a;
        }
        
        return 1 - bt * this.#betaCF(1 - x, b, a) / b;
    }

    #betaCF(x, a, b) {
        const eps = 1e-14;
        const maxIter = 100;
        
        let m = 1;
        let m2 = 0;
        let d = 1;
        let d2 = 0;
        let sum = 0;
        
        for (let i = 0; i <= maxIter; i++) {
            const aa = i * (b - i) * x / ((a + 2 * i - 1) * (a + 2 * i));
            d = 1 + aa * d2;
            if (Math.abs(d) < eps) d = eps;
            d = 1 / d;
            
            m = 1 + aa / m2;
            if (Math.abs(m) < eps) m = eps;
            
            const delta = m * d;
            sum = delta;
            
            if (Math.abs(delta - 1) < eps) break;
            
            m2 = m;
            d2 = d;
        }
        
        return sum;
    }

    #tCDF(x, df) {
        const t = x / Math.sqrt(df);
        const a = df / 2;
        const b = 0.5;
        return 1 - 0.5 * this.#betaIncomplete(df / (df + t * t), a, b);
    }

    #fCDF(x, df1, df2) {
        const p = this.#betaIncomplete(
            df1 * x / (df1 * x + df2),
            df1 / 2,
            df2 / 2
        );
        return 1 - p;
    }

    #chiSquareCDF(x, df) {
        return this.#gammaIncomplete(x / 2, df / 2);
    }

    #tCritical(df, alpha) {
        // Approximation valid for df > 0
        const a = 1 / (df - 0.5);
        const b = 48 / (df * df);
        const c = 96 / Math.pow(df, 3);
        const d = Math.sqrt(df / 2) * Math.exp(
            (Math.log(df / 2) + 1) * 0.5 +
            0.5 * Math.log(Math.PI) -
            df * 0.5
        );
        return (1 - alpha) * Math.sqrt(df) * (1 + a + b + c) / d;
    }

    #erf(x) {
        // Error function approximation
        const sign = x >= 0 ? 1 : -1;
        x = Math.abs(x);
        const a1 = 0.254829592;
        const a2 = -0.284496736;
        const a3 = 1.421413741;
        const a4 = -1.453152027;
        const a5 = 1.061405429;
        const p = 0.3275911;
        const t = 1 / (1 + p * x);
        const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return sign * y;
    }

    // 2. Probability Distributions
    uniformPDF(x, min = 0, max = 1) {
        if (x < min || x > max) return 0;
        return 1 / (max - min);
    }

    uniformCDF(x, min = 0, max = 1) {
        if (x < min) return 0;
        if (x > max) return 1;
        return (x - min) / (max - min);
    }

    normalPDF(x, mean = 0, stddev = 1) {
        const z = (x - mean) / stddev;
        return Math.exp(-0.5 * z * z) / (stddev * this.SQRT_2PI);
    }

    normalCDF(x, mean = 0, stddev = 1) {
        const z = (x - mean) / stddev;
        return 0.5 * (1 + this.#erf(z / Math.SQRT2));
    }

    binomialPMF(k, n, p) {
        if (k < 0 || k > n) return 0;
        return this.#combination(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k);
    }

    binomialCDF(k, n, p) {
        let sum = 0;
        for (let i = 0; i <= k; i++) {
            sum += this.binomialPMF(i, n, p);
        }
        return sum;
    }

    poissonPMF(k, lambda) {
        return Math.exp(-lambda) * Math.pow(lambda, k) / this.#factorial(k);
    }

    poissonCDF(k, lambda) {
        let sum = 0;
        for (let i = 0; i <= k; i++) {
            sum += this.poissonPMF(i, lambda);
        }
        return sum;
    }

    exponentialPDF(x, rate) {
        if (x < 0) return 0;
        return rate * Math.exp(-rate * x);
    }

    exponentialCDF(x, rate) {
        if (x < 0) return 0;
        return 1 - Math.exp(-rate * x);
    }

    geometricPMF(k, p) {
        if (k < 1) return 0;
        return p * Math.pow(1 - p, k - 1);
    }

    geometricCDF(k, p) {
        if (k < 1) return 0;
        return 1 - Math.pow(1 - p, k);
    }

    betaPDF(x, alpha, beta) {
        if (x < 0 || x > 1) return 0;
        return Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1) / this.#beta(alpha, beta);
    }

    gammaPDF(x, shape, scale = 1) {
        if (x < 0) return 0;
        return (Math.pow(x, shape - 1) * Math.exp(-x / scale)) / 
               (Math.pow(scale, shape) * this.#gamma(shape));
    }

    logNormalPDF(x, mu = 0, sigma = 1) {
        if (x <= 0) return 0;
        const logX = Math.log(x);
        return Math.exp(-Math.pow(logX - mu, 2) / (2 * sigma * sigma)) / 
               (x * sigma * this.SQRT_2PI);
    }

    bernoulliPMF(k, p) {
        if (k === 1) return p;
        if (k === 0) return 1 - p;
        return 0;
    }

    multinomialPMF(counts, probs) {
        if (counts.length !== probs.length) {
            throw new Error('Counts and probabilities arrays must have the same length');
        }
        const n = counts.reduce((a, b) => a + b, 0);
        const numerator = this.#factorial(n);
        const denominator = counts.reduce((acc, count) => acc * this.#factorial(count), 1);
        const probProduct = counts.reduce((acc, count, i) => acc * Math.pow(probs[i], count), 1);
        return (numerator / denominator) * probProduct;
    }

    // 1. Descriptive Statistics
    mean(arr) {
        this.#validateArray(arr);
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    median(arr) {
        this.#validateArray(arr);
        const sorted = this.#sortArray(arr);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
    }

    mode(arr) {
        this.#validateArray(arr);
        const counts = new Map();
        arr.forEach(val => counts.set(val, (counts.get(val) || 0) + 1));
        let maxCount = 0;
        let modes = [];
        counts.forEach((count, val) => {
            if (count > maxCount) {
                maxCount = count;
                modes = [val];
            } else if (count === maxCount) {
                modes.push(val);
            }
        });
        return modes;
    }

    variance(arr, sample = false) {
        this.#validateArray(arr);
        const mean = this.mean(arr);
        const n = sample ? arr.length - 1 : arr.length;
        return arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
    }

    stddev(arr, sample = false) {
        return Math.sqrt(this.variance(arr, sample));
    }

    range(arr) {
        this.#validateArray(arr);
        return Math.max(...arr) - Math.min(...arr);
    }

    iqr(arr) {
        const q = this.quartiles(arr);
        return q.q3 - q.q1;
    }

    skewness(arr) {
        this.#validateArray(arr);
        const mean = this.mean(arr);
        const stddev = this.stddev(arr);
        const n = arr.length;
        return arr.reduce((a, b) => a + Math.pow((b - mean) / stddev, 3), 0) / n;
    }

    kurtosis(arr) {
        this.#validateArray(arr);
        const mean = this.mean(arr);
        const stddev = this.stddev(arr);
        const n = arr.length;
        return arr.reduce((a, b) => a + Math.pow((b - mean) / stddev, 4), 0) / n;
    }

    quantile(arr, q) {
        this.#validateArray(arr);
        if (q < 0 || q > 1) {
            throw new Error('Quantile must be between 0 and 1');
        }
        const sorted = this.#sortArray(arr);
        const pos = (sorted.length - 1) * q;
        const base = Math.floor(pos);
        const rest = pos - base;
        if (sorted[base + 1] !== undefined) {
            return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
        } else {
            return sorted[base];
        }
    }

    quartiles(arr) {
        return {
            q1: this.quantile(arr, 0.25),
            q2: this.quantile(arr, 0.5),
            q3: this.quantile(arr, 0.75)
        };
    }

    coefficientOfVariation(arr) {
        this.#validateArray(arr);
        return this.stddev(arr) / this.mean(arr);
    }

    percentile(arr, p) {
        if (p < 0 || p > 100) {
            throw new Error('Percentile must be between 0 and 100');
        }
        return this.quantile(arr, p / 100);
    }

    // Summary statistics
    summary(arr) {
        this.#validateArray(arr);
        const sorted = this.#sortArray(arr);
        const q = this.quartiles(arr);
        const mean = this.mean(arr);
        
        return {
            n: arr.length,
            min: sorted[0],
            max: sorted[sorted.length - 1],
            range: this.range(arr),
            mean: mean,
            median: q.q2,
            mode: this.mode(arr),
            variance: this.variance(arr),
            stddev: this.stddev(arr),
            skewness: this.skewness(arr),
            kurtosis: this.kurtosis(arr),
            q1: q.q1,
            q3: q.q3,
            iqr: this.iqr(arr),
            cv: this.coefficientOfVariation(arr)
        };
    }

    // 3. Statistical Inference
    confidenceInterval(arr, confidence = 0.95) {
        this.#validateArray(arr);
        const n = arr.length;
        const mean = this.mean(arr);
        const stderr = this.stddev(arr, true) / Math.sqrt(n);
        const alpha = 1 - confidence;
        // Using t-distribution critical values for small samples
        const t = this.#tCritical(n - 1, alpha / 2);
        const margin = t * stderr;
        return {
            lower: mean - margin,
            upper: mean + margin,
            mean,
            margin
        };
    }

    tTest(arr, mu0 = 0, alternative = 'two-sided') {
        this.#validateArray(arr);
        const n = arr.length;
        const mean = this.mean(arr);
        const stderr = this.stddev(arr, true) / Math.sqrt(n);
        const t = (mean - mu0) / stderr;
        const df = n - 1;

        let pValue;
        if (alternative === 'two-sided') {
            pValue = 2 * (1 - this.#tCDF(Math.abs(t), df));
        } else if (alternative === 'greater') {
            pValue = 1 - this.#tCDF(t, df);
        } else if (alternative === 'less') {
            pValue = this.#tCDF(t, df);
        }

        return {
            statistic: t,
            pValue,
            df,
            mean,
            stderr
        };
    }

    tTestTwoSample(arr1, arr2, equalVariance = false) {
        this.#validateArray(arr1);
        this.#validateArray(arr2);
        
        const n1 = arr1.length;
        const n2 = arr2.length;
        const mean1 = this.mean(arr1);
        const mean2 = this.mean(arr2);
        const var1 = this.variance(arr1, true);
        const var2 = this.variance(arr2, true);
        
        let t, df;
        if (equalVariance) {
            const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
            const stderr = Math.sqrt(pooledVar * (1/n1 + 1/n2));
            t = (mean1 - mean2) / stderr;
            df = n1 + n2 - 2;
        } else {
            const stderr = Math.sqrt(var1/n1 + var2/n2);
            t = (mean1 - mean2) / stderr;
            // Welch-Satterthwaite approximation for df
            df = Math.pow(var1/n1 + var2/n2, 2) / 
                 (Math.pow(var1/n1, 2)/(n1-1) + Math.pow(var2/n2, 2)/(n2-1));
        }

        const pValue = 2 * (1 - this.#tCDF(Math.abs(t), df));
        
        return {
            statistic: t,
            pValue,
            df,
            mean1,
            mean2,
            var1,
            var2
        };
    }

    pairedTTest(arr1, arr2) {
        if (arr1.length !== arr2.length) {
            throw new Error('Arrays must have equal length for paired t-test');
        }
        const differences = arr1.map((x, i) => x - arr2[i]);
        return this.tTest(differences);
    }

    anova(...groups) {
        // One-way ANOVA
        const k = groups.length; // number of groups
        const n = groups.reduce((sum, group) => sum + group.length, 0); // total sample size
        
        // Calculate group means and overall mean
        const groupMeans = groups.map(group => this.mean(group));
        const overallMean = groups.flat().reduce((a, b) => a + b, 0) / n;
        
        // Calculate sum of squares
        const ssb = groups.reduce((sum, group, i) => 
            sum + group.length * Math.pow(groupMeans[i] - overallMean, 2), 0);
        
        const ssw = groups.reduce((sum, group, i) => 
            sum + group.reduce((s, x) => s + Math.pow(x - groupMeans[i], 2), 0), 0);
        
        const sst = ssb + ssw;
        
        // Degrees of freedom
        const dfb = k - 1;
        const dfw = n - k;
        const dft = n - 1;
        
        // Mean squares
        const msb = ssb / dfb;
        const msw = ssw / dfw;
        
        // F-statistic
        const f = msb / msw;
        
        // p-value using F-distribution
        const pValue = 1 - this.#fCDF(f, dfb, dfw);
        
        return {
            f,
            pValue,
            dfb,
            dfw,
            msb,
            msw,
            ssb,
            ssw,
            sst
        };
    }

    chiSquareTest(observed, expected) {
        if (observed.length !== expected.length) {
            throw new Error('Observed and expected arrays must have equal length');
        }
        
        const statistic = observed.reduce((sum, o, i) => {
            const e = expected[i];
            return sum + Math.pow(o - e, 2) / e;
        }, 0);
        
        const df = observed.length - 1;
        const pValue = 1 - this.#chiSquareCDF(statistic, df);
        
        return {
            statistic,
            pValue,
            df
        };
    }

    // Visualization methods
    visualize(type, data, elementId) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        if (type === 'histogram') {
            const values = data.values;
            const min = Math.min(...values);
            const max = Math.max(...values);
            const binCount = Math.ceil(Math.sqrt(values.length));
            const binWidth = (max - min) / binCount;
            
            const bins = Array(binCount).fill(0);
            values.forEach(val => {
                const binIndex = Math.min(Math.floor((val - min) / binWidth), binCount - 1);
                bins[binIndex]++;
            });
            
            const labels = bins.map((_, i) => (min + (i + 0.5) * binWidth).toFixed(2));
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels,
                    datasets: [{
                        label: data.label || 'Histogram',
                        data: bins,
                        backgroundColor: 'rgba(0, 123, 255, 0.5)',
                        borderColor: 'rgba(0, 123, 255, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Frequency'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Value'
                            }
                        }
                    }
                }
            });
        }
    }

    /**
     * Validates input array for statistical calculations
     * @private
     */
    _validateInput(arr, allowEmpty = false) {
        if (!Array.isArray(arr)) throw new Error('Input must be an array');
        if (!allowEmpty && arr.length === 0) throw new Error('Array cannot be empty');
        if (!arr.every(n => typeof n === 'number' && !isNaN(n))) {
            throw new Error('All elements must be valid numbers');
        }
    }

    /**
     * Sorts an array numerically
     * @private
     */
    _sort(arr) {
        return [...arr].sort((a, b) => a - b);
    }

    /**
     * Calculates the arithmetic mean
     * @param {number[]} arr - Array of numbers
     * @returns {number} Arithmetic mean
     */
    mean(arr) {
        this._validateInput(arr);
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    /**
     * Calculates the geometric mean
     * @param {number[]} arr - Array of positive numbers
     * @returns {number} Geometric mean
     */
    geometricMean(arr) {
        this._validateInput(arr);
        if (arr.some(x => x <= 0)) throw new Error('All numbers must be positive for geometric mean');
        return Math.pow(arr.reduce((a, b) => a * b, 1), 1 / arr.length);
    }

    /**
     * Calculates the harmonic mean
     * @param {number[]} arr - Array of non-zero numbers
     * @returns {number} Harmonic mean
     */
    harmonicMean(arr) {
        this._validateInput(arr);
        if (arr.some(x => x === 0)) throw new Error('Cannot calculate harmonic mean with zero values');
        return arr.length / arr.reduce((a, b) => a + 1/b, 0);
    }

    /**
     * Calculates the median
     * @param {number[]} arr - Array of numbers
     * @returns {number} Median value
     */
    median(arr) {
        this._validateInput(arr);
        const sorted = this._sort(arr);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid];
    }

    /**
     * Finds the mode(s) of the dataset
     * @param {number[]} arr - Array of numbers
     * @returns {number[]} Mode(s) of the dataset
     */
    mode(arr) {
        this._validateInput(arr);
        const counts = new Map();
        arr.forEach(n => counts.set(n, (counts.get(n) || 0) + 1));
        
        let maxCount = 0;
        let modes = [];
        
        counts.forEach((count, num) => {
            if (count > maxCount) {
                maxCount = count;
                modes = [num];
            } else if (count === maxCount) {
                modes.push(num);
            }
        });
        
        return modes;
    }

    /**
     * Calculates variance
     * @param {number[]} arr - Array of numbers
     * @param {boolean} [sample=false] - If true, calculates sample variance
     * @returns {number} Variance
     */
    variance(arr, sample = false) {
        this._validateInput(arr);
        const mu = this.mean(arr);
        const squaredDiffs = arr.map(x => Math.pow(x - mu, 2));
        return squaredDiffs.reduce((a, b) => a + b, 0) / (arr.length - (sample ? 1 : 0));
    }

    /**
     * Calculates standard deviation
     * @param {number[]} arr - Array of numbers
     * @param {boolean} [sample=false] - If true, calculates sample standard deviation
     * @returns {number} Standard deviation
     */
    stddev(arr, sample = false) {
        return Math.sqrt(this.variance(arr, sample));
    }

    /**
     * Calculates quartiles
     * @param {number[]} arr - Array of numbers
     * @returns {{q1: number, q2: number, q3: number}} Quartile values
     */
    quartiles(arr) {
        this._validateInput(arr);
        const sorted = this._sort(arr);
        const mid = Math.floor(sorted.length / 2);
        
        const lowerHalf = sorted.slice(0, mid);
        const upperHalf = sorted.length % 2 === 0 
            ? sorted.slice(mid)
            : sorted.slice(mid + 1);
        
        return {
            q1: this.median(lowerHalf),
            q2: this.median(sorted),
            q3: this.median(upperHalf)
        };
    }

    /**
     * Calculates the interquartile range (IQR)
     * @param {number[]} arr - Array of numbers
     * @returns {number} IQR value
     */
    iqr(arr) {
        const { q1, q3 } = this.quartiles(arr);
        return q3 - q1;
    }

    /**
     * Identifies outliers using the 1.5 * IQR rule
     * @param {number[]} arr - Array of numbers
     * @returns {{outliers: number[], bounds: {lower: number, upper: number}}}
     */
    outliers(arr) {
        this._validateInput(arr);
        const { q1, q3 } = this.quartiles(arr);
        const iqr = q3 - q1;
        const bounds = {
            lower: q1 - 1.5 * iqr,
            upper: q3 + 1.5 * iqr
        };
        
        return {
            outliers: arr.filter(x => x < bounds.lower || x > bounds.upper),
            bounds
        };
    }

    /**
     * Calculates skewness (third standardized moment)
     * @param {number[]} arr - Array of numbers
     * @returns {number} Skewness value
     */
    skewness(arr) {
        this._validateInput(arr);
        const mu = this.mean(arr);
        const sigma = this.stddev(arr);
        const n = arr.length;
        
        const cubedDiffs = arr.map(x => Math.pow((x - mu) / sigma, 3));
        return (n / ((n-1) * (n-2))) * cubedDiffs.reduce((a, b) => a + b, 0);
    }

    /**
     * Calculates kurtosis (fourth standardized moment)
     * @param {number[]} arr - Array of numbers
     * @returns {number} Kurtosis value
     */
    kurtosis(arr) {
        this._validateInput(arr);
        const mu = this.mean(arr);
        const sigma = this.stddev(arr);
        const n = arr.length;
        
        const fourthMoment = arr.map(x => Math.pow((x - mu) / sigma, 4))
            .reduce((a, b) => a + b, 0);
            
        return ((n * (n+1)) / ((n-1) * (n-2) * (n-3))) * fourthMoment - 
               (3 * (n-1) * (n-1)) / ((n-2) * (n-3));
    }

    /**
     * Calculates correlation coefficient between two arrays
     * @param {number[]} arr1 - First array of numbers
     * @param {number[]} arr2 - Second array of numbers
     * @returns {number} Correlation coefficient
     */
    correlation(arr1, arr2) {
        if (arr1.length !== arr2.length) {
            throw new Error('Arrays must have the same length');
        }
        this._validateInput(arr1);
        this._validateInput(arr2);
        
        const mean1 = this.mean(arr1);
        const mean2 = this.mean(arr2);
        const std1 = this.stddev(arr1);
        const std2 = this.stddev(arr2);
        
        const n = arr1.length;
        let sum = 0;
        
        for (let i = 0; i < n; i++) {
            sum += ((arr1[i] - mean1) / std1) * ((arr2[i] - mean2) / std2);
        }
        
        return sum / (n - 1);
    }

    /**
     * Calculates summary statistics
     * @param {number[]} arr - Array of numbers
     * @returns {Object} Summary statistics
     */
    summary(arr) {
        this._validateInput(arr);
        const sorted = this._sort(arr);
        const quarts = this.quartiles(arr);
        
        return {
            n: arr.length,
            min: sorted[0],
            max: sorted[sorted.length - 1],
            mean: this.mean(arr),
            median: quarts.q2,
            q1: quarts.q1,
            q3: quarts.q3,
            iqr: this.iqr(arr),
            variance: this.variance(arr),
            stddev: this.stddev(arr),
            skewness: this.skewness(arr),
            kurtosis: this.kurtosis(arr)
        };
    }

    /**
     * Creates a Chart.js visualization if Chart.js is available
     * @param {string} type - Chart type ('histogram', 'box', 'scatter')
     * @param {Object} data - Data configuration
     * @param {string} targetId - Target canvas element ID
     * @returns {Object|null} Chart instance or null if Chart.js is not available
     */
    visualize(type, data, targetId) {
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js is required for visualization. Include it in your HTML:');
            console.warn('<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>');
            return null;
        }

        const canvas = document.getElementById(targetId);
        if (!canvas) {
            throw new Error(`Canvas element with id '${targetId}' not found`);
        }

        const ctx = canvas.getContext('2d');
        let chartConfig;

        switch (type) {
            case 'histogram':
                const bins = this._createHistogramBins(data.values);
                chartConfig = {
                    type: 'bar',
                    data: {
                        labels: bins.map(b => `${b.start.toFixed(2)}-${b.end.toFixed(2)}`),
                        datasets: [{
                            label: data.label || 'Frequency',
                            data: bins.map(b => b.count),
                            backgroundColor: 'rgba(0, 102, 204, 0.5)',
                            borderColor: 'rgba(0, 102, 204, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                };
                break;

            case 'box':
                const stats = this.summary(data.values);
                chartConfig = {
                    type: 'boxplot',
                    data: {
                        labels: [data.label || 'Distribution'],
                        datasets: [{
                            label: 'Box Plot',
                            data: [{
                                min: stats.min,
                                q1: stats.q1,
                                median: stats.median,
                                q3: stats.q3,
                                max: stats.max,
                                outliers: this.outliers(data.values).outliers
                            }]
                        }]
                    },
                    options: {
                        responsive: true
                    }
                };
                break;

            case 'scatter':
                if (!data.x || !data.y) {
                    throw new Error('Scatter plot requires x and y data arrays');
                }
                chartConfig = {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: data.label || 'Scatter Plot',
                            data: data.x.map((x, i) => ({x, y: data.y[i]})),
                            backgroundColor: 'rgba(0, 102, 204, 0.5)'
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: { type: 'linear', position: 'bottom' }
                        }
                    }
                };
                break;

            default:
                throw new Error(`Unsupported chart type: ${type}`);
        }

        return new Chart(ctx, chartConfig);
    }

    /**
     * Creates histogram bins for visualization
     * @private
     */
    _createHistogramBins(arr, binCount = 10) {
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        const binWidth = (max - min) / binCount;
        const bins = Array(binCount).fill().map((_, i) => ({
            start: min + i * binWidth,
            end: min + (i + 1) * binWidth,
            count: 0
        }));

        arr.forEach(val => {
            const binIndex = Math.min(
                Math.floor((val - min) / binWidth),
                binCount - 1
            );
            bins[binIndex].count++;
        });

        return bins;
    }

    // Advanced Statistical Methods
    
    /**
     * Performs a one-way ANOVA test
     * @param {...Array} groups - Two or more groups of numbers to compare
     * @returns {Object} ANOVA test results including F-statistic, p-value, and effect size
     */
    anova(...groups) {
        // Validate input
        if (groups.length < 2) {
            throw new Error('ANOVA requires at least two groups');
        }
        
        // Calculate group means and overall mean
        const groupMeans = groups.map(g => this.mean(g));
        const allValues = groups.flat();
        const overallMean = this.mean(allValues);
        
        // Calculate Sum of Squares Between groups (SSB)
        const ssb = groups.reduce((sum, group, i) => {
            return sum + group.length * Math.pow(groupMeans[i] - overallMean, 2);
        }, 0);
        
        // Calculate Sum of Squares Within groups (SSW)
        const ssw = groups.reduce((sum, group, i) => {
            return sum + group.reduce((s, value) => {
                return s + Math.pow(value - groupMeans[i], 2);
            }, 0);
        }, 0);
        
        // Calculate degrees of freedom
        const dfb = groups.length - 1; // between groups
        const dfw = allValues.length - groups.length; // within groups
        
        // Calculate Mean Squares
        const msb = ssb / dfb;
        const msw = ssw / dfw;
        
        // Calculate F-statistic
        const f = msb / msw;
        
        // Calculate p-value using F-distribution approximation
        const pValue = this._calculateFPValue(f, dfb, dfw);
        
        // Calculate effect size (η² - eta squared)
        const etaSquared = ssb / (ssb + ssw);
        
        return {
            f_statistic: f,
            p_value: pValue,
            effect_size: etaSquared,
            df_between: dfb,
            df_within: dfw,
            mean_square_between: msb,
            mean_square_within: msw,
            groups_summary: groups.map((g, i) => ({
                n: g.length,
                mean: groupMeans[i],
                variance: this.variance(g)
            }))
        };
    }

    /**
     * Performs a t-test between two groups
     * @param {Array} group1 - First group of numbers
     * @param {Array} group2 - Second group of numbers
     * @param {Object} options - Test options (paired, twoTailed)
     * @returns {Object} T-test results including t-statistic, p-value, and effect size
     */
    tTest(group1, group2, options = { paired: false, twoTailed: true }) {
        if (options.paired && group1.length !== group2.length) {
            throw new Error('Paired t-test requires equal group sizes');
        }

        let t, df;
        
        if (options.paired) {
            // Paired t-test
            const differences = group1.map((v, i) => v - group2[i]);
            const meanDiff = this.mean(differences);
            const sdDiff = this.stddev(differences);
            t = meanDiff / (sdDiff / Math.sqrt(group1.length));
            df = group1.length - 1;
        } else {
            // Independent t-test
            const n1 = group1.length;
            const n2 = group2.length;
            const mean1 = this.mean(group1);
            const mean2 = this.mean(group2);
            const var1 = this.variance(group1);
            const var2 = this.variance(group2);
            
            // Pooled standard error
            const pooledSD = Math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));
            const stderr = Math.sqrt(pooledSD * (1/n1 + 1/n2));
            t = (mean1 - mean2) / stderr;
            
            // Welch-Satterthwaite approximation for df
            df = Math.pow((var1/n1 + var2/n2), 2) / 
                 (Math.pow(var1/n1, 2)/(n1-1) + Math.pow(var2/n2, 2)/(n2-1));
        }
        
        // Calculate p-value
        const pValue = this._calculateTPValue(t, df, options.twoTailed);
        
        // Calculate Cohen's d effect size
        const d = this._calculateCohensD(group1, group2, options.paired);
        
        return {
            t_statistic: t,
            p_value: pValue,
            degrees_of_freedom: df,
            effect_size: d,
            effect_size_type: "Cohen's d",
            test_type: options.paired ? 'Paired' : 'Independent',
            groups_summary: [
                {
                    n: group1.length,
                    mean: this.mean(group1),
                    std_dev: this.stddev(group1)
                },
                {
                    n: group2.length,
                    mean: this.mean(group2),
                    std_dev: this.stddev(group2)
                }
            ]
        };
    }

    /**
     * Calculates the confidence interval for a mean
     * @param {Array} data - Array of numbers
     * @param {number} confidence - Confidence level (0-1), default 0.95
     * @returns {Object} Confidence interval bounds and details
     */
    confidenceInterval(data, confidence = 0.95) {
        const mean = this.mean(data);
        const sd = this.stddev(data, true); // sample standard deviation
        const n = data.length;
        const se = sd / Math.sqrt(n);
        const alpha = 1 - confidence;
        // Using t-distribution critical values for small samples
        const t = this.#tCritical(n - 1, alpha / 2);
        const margin = t * se;
        
        return {
            mean,
            lower_bound: mean - margin,
            upper_bound: mean + margin,
            confidence_level: confidence,
            standard_error: se,
            margin_of_error: margin,
            sample_size: n
        };
    }

    /**
     * Performs a chi-square test of independence
     * @param {Array<Array<number>>} contingencyTable - 2D array representing contingency table
     * @returns {Object} Chi-square test results
     */
    chiSquareTest(contingencyTable) {
        // Calculate row and column totals
        const rowTotals = contingencyTable.map(row => row.reduce((a, b) => a + b, 0));
        const colTotals = contingencyTable[0].map((_, i) => 
            contingencyTable.reduce((sum, row) => sum + row[i], 0));
        const total = rowTotals.reduce((a, b) => a + b, 0);

        // Calculate expected frequencies and chi-square statistic
        let chiSquare = 0;
        const expected = [];
        
        for (let i = 0; i < contingencyTable.length; i++) {
            expected[i] = [];
            for (let j = 0; j < contingencyTable[0].length; j++) {
                const expectedValue = (rowTotals[i] * colTotals[j]) / total;
                expected[i][j] = expectedValue;
                const observed = contingencyTable[i][j];
                chiSquare += Math.pow(observed - expectedValue, 2) / expectedValue;
            }
        }

        // Calculate degrees of freedom
        const df = (contingencyTable.length - 1) * (contingencyTable[0].length - 1);
        
        // Calculate p-value
        const pValue = 1 - this.#chiSquareCDF(chiSquare, df);
        
        // Calculate Cramer's V effect size
        const cramersV = Math.sqrt(chiSquare / (total * Math.min(contingencyTable.length - 1, contingencyTable[0].length - 1)));

        return {
            chi_square: chiSquare,
            p_value: pValue,
            degrees_of_freedom: df,
            effect_size: cramersV,
            effect_size_type: "Cramer's V",
            observed: contingencyTable,
            expected: expected,
            row_totals: rowTotals,
            column_totals: colTotals
        };
    }

    // Helper methods for statistical distributions

    _calculateFPValue(f, df1, df2) {
        // Approximation of F-distribution p-value
        // Using Wilson-Hilferty transformation
        const z = ((f/df2)**(1/3) - (1 - 2/(9*df2)))/(Math.sqrt(2/(9*df2)));
        return 1 - this._normalCDF(z);
    }

    _calculateTPValue(t, df, twoTailed = true) {
        // Approximation of t-distribution p-value
        // Using standard normal approximation for large df
        let p;
        if (df > 30) {
            p = 1 - this._normalCDF(Math.abs(t));
        } else {
            // Student's t-distribution approximation for small df
            p = 1 - this._studentTCDF(Math.abs(t), df);
        }
        return twoTailed ? 2 * p : p;
    }

    _calculateChiSquarePValue(chiSquare, df) {
        // Approximation of chi-square distribution p-value
        // Using Wilson-Hilferty transformation
        const z = Math.sqrt(2 * chiSquare) - Math.sqrt(2 * df - 1);
        return 1 - this._normalCDF(z);
    }

    _normalCDF(x) {
        // Approximation of the standard normal cumulative distribution function
        const t = 1 / (1 + 0.2316419 * Math.abs(x));
        const d = 0.3989423 * Math.exp(-x * x / 2);
        const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - p : p;
    }

    _studentTCDF(t, df) {
        // Approximation of Student's t cumulative distribution function
        const x = df / (df + t * t);
        return 1 - 0.5 * this._incompleteBeta(x, df/2, 0.5);
    }

    _incompleteBeta(x, a, b) {
        // Approximation of incomplete beta function
        // Using continued fraction expansion
        const maxIterations = 100;
        const epsilon = 1e-8;
        
        if (x === 0) return 0;
        if (x === 1) return 1;
        
        const lnBeta = this._logGamma(a) + this._logGamma(b) - this._logGamma(a + b);
        const lnX = Math.log(x);
        const lnY = Math.log(1 - x);
        
        let h = Math.exp(a * lnX + b * lnY - lnBeta);
        let t = h;
        let n = 1;
        let cn = 1;
        
        while (n < maxIterations && Math.abs(cn) > epsilon) {
            const aold = a + 2 * n - 2;
            cn = (a + n - 1) * (b - n) * x / ((aold) * (aold + 1));
            h = h * cn + 1;
            t = t * cn;
            n++;
        }
        
        return h / a;
    }

    _logGamma(x) {
        // Approximation of natural logarithm of gamma function
        const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
                  -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
        let sum = 1.000000000190015;
        for (let i = 0; i < 6; i++) {
            sum += c[i] / (x + i + 1);
        }
        return (x + 0.5) * Math.log(x + 5.5) - (x + 5.5) + Math.log(2.5066282746310005 * sum / x);
    }

    _getTCriticalValue(df, alpha) {
        // Approximation of t-distribution critical value
        // Using inverse of Student's t CDF
        const p = 1 - alpha;
        const t = Math.sqrt(df) * (Math.pow(p, -2/df) - 1) / Math.sqrt(2);
        return t;
    }

    _calculateCohensD(group1, group2, paired = false) {
        if (paired) {
            const differences = group1.map((v, i) => v - group2[i]);
            const meanDiff = this.mean(differences);
            const sdDiff = this.stddev(differences);
            return meanDiff / sdDiff;
        } else {
            const n1 = group1.length;
            const n2 = group2.length;
            const mean1 = this.mean(group1);
            const mean2 = this.mean(group2);
            const var1 = this.variance(group1);
            const var2 = this.variance(group2);
            
            // Pooled standard deviation
            const pooledSD = Math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));
            return (mean1 - mean2) / pooledSD;
        }
    }

    // 4. Regression Analysis
    linearRegression(x, y) {
        if (x.length !== y.length) {
            throw new Error('Input arrays must have equal length');
        }
        this.#validateArray(x);
        this.#validateArray(y);

        const n = x.length;
        const meanX = this.mean(x);
        const meanY = this.mean(y);

        // Calculate coefficients
        let numerator = 0;
        let denominator = 0;
        for (let i = 0; i < n; i++) {
            numerator += (x[i] - meanX) * (y[i] - meanY);
            denominator += Math.pow(x[i] - meanX, 2);
        }

        const slope = numerator / denominator;
        const intercept = meanY - slope * meanX;

        // Calculate R-squared
        const yPred = x.map(xi => slope * xi + intercept);
        const rSquared = this.rSquared(y, yPred);

        // Calculate standard errors
        const mse = this.meanSquaredError(y, yPred);
        const slopeStdErr = Math.sqrt(mse / denominator);
        const interceptStdErr = Math.sqrt(mse * (1/n + Math.pow(meanX, 2)/denominator));

        // Calculate t-statistics and p-values
        const df = n - 2;
        const slopeTStat = slope / slopeStdErr;
        const interceptTStat = intercept / interceptStdErr;
        const slopePValue = 2 * (1 - this.#tCDF(Math.abs(slopeTStat), df));
        const interceptPValue = 2 * (1 - this.#tCDF(Math.abs(interceptTStat), df));

        return {
            slope,
            intercept,
            rSquared,
            slopeStdErr,
            interceptStdErr,
            slopeTStat,
            interceptTStat,
            slopePValue,
            interceptPValue,
            df,
            mse,
            predict: (newX) => newX.map(xi => slope * xi + intercept)
        };
    }

    polynomialRegression(x, y, degree = 2) {
        if (x.length !== y.length) {
            throw new Error('Input arrays must have equal length');
        }
        this.#validateArray(x);
        this.#validateArray(y);
        
        const n = x.length;
        const X = new Array(n);
        
        // Create design matrix
        for (let i = 0; i < n; i++) {
            X[i] = new Array(degree + 1);
            for (let j = 0; j <= degree; j++) {
                X[i][j] = Math.pow(x[i], j);
            }
        }
        
        // Solve normal equations using matrix operations
        const Xt = this.#transposeMatrix(X);
        const XtX = this.#multiplyMatrices(Xt, X);
        const XtY = this.#multiplyMatrices(Xt, y.map(yi => [yi]));
        const coefficients = this.#solveLinearSystem(XtX, XtY).map(c => c[0]);
        
        // Calculate predicted values
        const yPred = x.map(xi => {
            let sum = 0;
            for (let j = 0; j <= degree; j++) {
                sum += coefficients[j] * Math.pow(xi, j);
            }
            return sum;
        });
        
        // Calculate R-squared and MSE
        const rSquared = this.rSquared(y, yPred);
        const mse = this.meanSquaredError(y, yPred);
        
        return {
            coefficients,
            rSquared,
            mse,
            predict: (newX) => newX.map(xi => {
                let sum = 0;
                for (let j = 0; j <= degree; j++) {
                    sum += coefficients[j] * Math.pow(xi, j);
                }
                return sum;
            })
        };
    }

    logisticRegression(x, y, maxIter = 100, learningRate = 0.1, tolerance = 1e-6) {
        if (x.length !== y.length) {
            throw new Error('Input arrays must have equal length');
        }
        this.#validateArray(x);
        if (!y.every(yi => yi === 0 || yi === 1)) {
            throw new Error('Y values must be binary (0 or 1)');
        }

        let b0 = 0; // intercept
        let b1 = 0; // slope
        let prevLogLikelihood = -Infinity;

        for (let iter = 0; iter < maxIter; iter++) {
            // Calculate predictions
            const preds = x.map(xi => this.#sigmoid(b0 + b1 * xi));

            // Calculate gradients
            let gradB0 = 0;
            let gradB1 = 0;
            for (let i = 0; i < x.length; i++) {
                gradB0 += y[i] - preds[i];
                gradB1 += (y[i] - preds[i]) * x[i];
            }

            // Update parameters
            b0 += learningRate * gradB0;
            b1 += learningRate * gradB1;

            // Check convergence using log-likelihood
            const logLikelihood = x.reduce((sum, xi, i) => {
                const p = this.#sigmoid(b0 + b1 * xi);
                return sum + (y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
            }, 0);

            if (Math.abs(logLikelihood - prevLogLikelihood) < tolerance) {
                break;
            }
            prevLogLikelihood = logLikelihood;
        }

        // Calculate metrics
        const preds = x.map(xi => this.#sigmoid(b0 + b1 * xi));
        const threshold = 0.5;
        const yPred = preds.map(p => p >= threshold ? 1 : 0);
        const accuracy = y.reduce((sum, yi, i) => sum + (yi === yPred[i] ? 1 : 0), 0) / y.length;

        return {
            intercept: b0,
            slope: b1,
            accuracy,
            predict: (newX) => newX.map(xi => this.#sigmoid(b0 + b1 * xi)),
            predictClass: (newX) => newX.map(xi => this.#sigmoid(b0 + b1 * xi) >= threshold ? 1 : 0)
        };
    }

    // Helper methods for regression
    rSquared(yTrue, yPred) {
        const meanY = this.mean(yTrue);
        const ssTotal = yTrue.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0);
        const ssResidual = this.meanSquaredError(yTrue, yPred) * yTrue.length;
        return 1 - (ssResidual / ssTotal);
    }

    meanSquaredError(yTrue, yPred) {
        if (yTrue.length !== yPred.length) {
            throw new Error('Arrays must have equal length');
        }
        return yTrue.reduce((sum, yi, i) => sum + Math.pow(yi - yPred[i], 2), 0) / yTrue.length;
    }

    #sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    #transposeMatrix(matrix) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = Array(cols).fill().map(() => Array(rows));
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }

    #multiplyMatrices(a, b) {
        const rowsA = a.length;
        const colsA = a[0].length;
        const rowsB = b.length;
        const colsB = b[0].length;
        
        if (colsA !== rowsB) {
            throw new Error('Invalid matrix dimensions for multiplication');
        }
        
        const result = Array(rowsA).fill().map(() => Array(colsB).fill(0));
        
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                for (let k = 0; k < colsA; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }

    #solveLinearSystem(A, b) {
        // Gaussian elimination with partial pivoting
        const n = A.length;
        const x = new Array(n).fill(0);
        const L = Array(n).fill().map(() => Array(n).fill(0));
        const U = Array(n).fill().map(() => Array(n).fill(0));

        // LU decomposition
        for (let i = 0; i < n; i++) {
            // Upper triangular
            for (let k = i; k < n; k++) {
                let sum = 0;
                for (let j = 0; j < i; j++) {
                    sum += L[i][j] * U[j][k];
                }
                U[i][k] = A[i][k] - sum;
            }

            // Lower triangular
            for (let k = i; k < n; k++) {
                if (i === k) {
                    L[i][i] = 1;
                } else {
                    let sum = 0;
                    for (let j = 0; j < i; j++) {
                        sum += L[k][j] * U[j][i];
                    }
                    L[k][i] = (A[k][i] - sum) / U[i][i];
                }
            }
        }

        // Forward substitution (Ly = b)
        const y = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = 0; j < i; j++) {
                sum += L[i][j] * y[j];
            }
            y[i] = b[i] - sum;
        }

        // Backward substitution (Ux = y)
        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) {
                sum += U[i][j] * x[j];
            }
            x[i] = (y[i] - sum) / U[i][i];
        }

        return x;
    }

    // 5. Correlation Analysis
    correlation(x, y) {
        if (x.length !== y.length) {
            throw new Error('Arrays must have equal length');
        }
        this.#validateArray(x);
        this.#validateArray(y);

        const n = x.length;
        const meanX = this.mean(x);
        const meanY = this.mean(y);
        const sdX = this.stddev(x);
        const sdY = this.stddev(y);

        let sum = 0;
        for (let i = 0; i < n; i++) {
            sum += ((x[i] - meanX) / sdX) * ((y[i] - meanY) / sdY);
        }

        const r = sum / (n - 1);
        const rSquared = r * r;
        const t = r * Math.sqrt((n - 2) / (1 - rSquared));
        const df = n - 2;
        const pValue = 2 * (1 - this.#tCDF(Math.abs(t), df));

        return {
            coefficient: r,
            rSquared,
            t,
            pValue,
            df
        };
    }

    spearmanCorrelation(x, y) {
        if (x.length !== y.length) {
            throw new Error('Arrays must have equal length');
        }
        this.#validateArray(x);
        this.#validateArray(y);

        // Convert to ranks
        const xRanks = this.#calculateRanks(x);
        const yRanks = this.#calculateRanks(y);

        // Calculate correlation of ranks
        return this.correlation(xRanks, yRanks);
    }

    kendallCorrelation(x, y) {
        if (x.length !== y.length) {
            throw new Error('Arrays must have equal length');
        }
        this.#validateArray(x);
        this.#validateArray(y);

        const n = x.length;
        let concordant = 0;
        let discordant = 0;

        for (let i = 0; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) {
                const xDiff = x[i] - x[j];
                const yDiff = y[i] - y[j];
                if (xDiff * yDiff > 0) concordant++;
                else if (xDiff * yDiff < 0) discordant++;
            }
        }

        const tau = (concordant - discordant) / (0.5 * n * (n - 1));
        const z = 3 * tau * Math.sqrt(n * (n - 1)) / Math.sqrt(2 * (2 * n + 5));
        const pValue = 2 * (1 - this.#normalCDF(Math.abs(z)));

        return {
            coefficient: tau,
            z,
            pValue
        };
    }

    // Helper method for Kendall correlation
    #normalCDF(x) {
        return 0.5 * (1 + this.#erf(x / Math.SQRT2));
    }

    // 6. Time Series Analysis
    movingAverage(data, window) {
        this.#validateArray(data);
        if (window < 1 || window > data.length) {
            throw new Error('Window size must be between 1 and data length');
        }

        const result = new Array(data.length - window + 1);
        let sum = 0;

        // Initialize the first window
        for (let i = 0; i < window; i++) {
            sum += data[i];
        }
        result[0] = sum / window;

        // Slide the window
        for (let i = window; i < data.length; i++) {
            sum = sum - data[i - window] + data[i];
            result[i - window + 1] = sum / window;
        }

        return result;
    }

    exponentialSmoothing(data, alpha = 0.2) {
        this.#validateArray(data);
        if (alpha < 0 || alpha > 1) {
            throw new Error('Alpha must be between 0 and 1');
        }

        const result = new Array(data.length);
        result[0] = data[0];

        for (let i = 1; i < data.length; i++) {
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1];
        }

        return result;
    }

    doubleExponentialSmoothing(data, alpha = 0.2, beta = 0.1) {
        this.#validateArray(data);
        if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1) {
            throw new Error('Alpha and beta must be between 0 and 1');
        }

        const level = new Array(data.length);
        const trend = new Array(data.length);
        const result = new Array(data.length);

        // Initialize
        level[0] = data[0];
        trend[0] = data[1] - data[0];
        result[0] = level[0];

        // Calculate level, trend and forecast
        for (let i = 1; i < data.length; i++) {
            const prevLevel = level[i - 1];
            const prevTrend = trend[i - 1];

            // Update level and trend
            level[i] = alpha * data[i] + (1 - alpha) * (prevLevel + prevTrend);
            trend[i] = beta * (level[i] - prevLevel) + (1 - beta) * prevTrend;

            // Calculate forecast
            result[i] = level[i] + trend[i];
        }

        // Calculate predictions
        const predictions = new Array(data.length);
        const lastLevel = level[level.length - 1];
        const lastTrend = trend[trend.length - 1];

        for (let i = 0; i < data.length; i++) {
            predictions[i] = lastLevel + (i + 1) * lastTrend;
        }

        return {
            forecast: result,
            level,
            trend,
            predict: (steps) => {
                const predictions = new Array(steps);
                const lastLevel = level[level.length - 1];
                const lastTrend = trend[trend.length - 1];

                for (let i = 0; i < steps; i++) {
                    predictions[i] = lastLevel + (i + 1) * lastTrend;
                }

                return predictions;
            }
        };
    }

    seasonalDecomposition(data, period) {
        this.#validateArray(data);
        if (period < 2 || period > data.length / 2) {
            throw new Error('Period must be between 2 and half the data length');
        }

        const n = data.length;
        const numSeasons = Math.floor(n / period);

        // Calculate trend using centered moving average
        const trend = new Array(n).fill(null);
        if (period % 2 === 1) {
            // Odd period
            const halfPeriod = Math.floor(period / 2);
            for (let i = halfPeriod; i < n - halfPeriod; i++) {
                let sum = 0;
                for (let j = -halfPeriod; j <= halfPeriod; j++) {
                    sum += data[i + j];
                }
                trend[i] = sum / period;
            }
        } else {
            // Even period
            const halfPeriod = period / 2;
            for (let i = halfPeriod; i < n - halfPeriod; i++) {
                let sum = 0;
                for (let j = -halfPeriod; j < halfPeriod; j++) {
                    sum += data[i + j];
                }
                trend[i] = sum / period;
            }
        }

        // Calculate seasonal pattern
        const seasonal = new Array(period).fill(0);
        const seasonalCounts = new Array(period).fill(0);

        for (let i = 0; i < n; i++) {
            if (trend[i] !== null) {
                const seasonIndex = i % period;
                seasonal[seasonIndex] += data[i] / trend[i];
                seasonalCounts[seasonIndex]++;
            }
        }

        // Normalize seasonal factors
        for (let i = 0; i < period; i++) {
            if (seasonalCounts[i] > 0) {
                seasonal[i] /= seasonalCounts[i];
            }
        }

        // Adjust seasonal factors to average to 1
        const seasonalMean = this.mean(seasonal);
        const normalizedSeasonal = seasonal.map(x => x / seasonalMean);

        // Calculate full seasonal component and residuals
        const seasonalComponent = new Array(n);
        const residuals = new Array(n);
        const fitted = new Array(n);

        for (let i = 0; i < n; i++) {
            if (trend[i] !== null) {
                seasonalComponent[i] = normalizedSeasonal[i % period];
                fitted[i] = trend[i] * seasonalComponent[i];
                residuals[i] = data[i] - fitted[i];
            } else {
                seasonalComponent[i] = null;
                fitted[i] = null;
                residuals[i] = null;
            }
        }

        return {
            trend,
            seasonal: normalizedSeasonal,
            seasonalComponent,
            residuals,
            fitted
        };
    }

    autocorrelation(data, lag = 1) {
        this.#validateArray(data);
        if (lag < 1 || lag >= data.length) {
            throw new Error('Lag must be between 1 and data length - 1');
        }

        const n = data.length;
        const mean = this.mean(data);
        let numerator = 0;
        let denominator = 0;

        for (let i = 0; i < n - lag; i++) {
            numerator += (data[i] - mean) * (data[i + lag] - mean);
        }

        for (let i = 0; i < n; i++) {
            denominator += Math.pow(data[i] - mean, 2);
        }

        return numerator / denominator;
    }

    autocorrelationFunction(data, maxLag = null) {
        this.#validateArray(data);
        if (!maxLag) maxLag = Math.floor(data.length / 4);
        if (maxLag < 1 || maxLag >= data.length) {
            throw new Error('Maximum lag must be between 1 and data length - 1');
        }

        const acf = new Array(maxLag + 1);
        acf[0] = 1; // Correlation with itself is always 1

        for (let lag = 1; lag <= maxLag; lag++) {
            acf[lag] = this.autocorrelation(data, lag);
        }

        // Calculate confidence intervals (±1.96/√n)
        const ci = 1.96 / Math.sqrt(data.length);

        return {
            acf,
            upperCI: Array(maxLag + 1).fill(ci),
            lowerCI: Array(maxLag + 1).fill(-ci)
        };
    }

    partialAutocorrelation(data, maxLag = null) {
        this.#validateArray(data);
        if (!maxLag) maxLag = Math.floor(data.length / 4);
        if (maxLag < 1 || maxLag >= data.length) {
            throw new Error('Maximum lag must be between 1 and data length - 1');
        }

        const pacf = new Array(maxLag + 1);
        pacf[0] = 1;

        // Durbin-Levinson algorithm
        const phi = Array(maxLag + 1).fill().map(() => Array(maxLag + 1).fill(0));
        const acf = this.autocorrelationFunction(data, maxLag).acf;

        for (let k = 1; k <= maxLag; k++) {
            // Calculate partial autocorrelation
            let numerator = acf[k];
            let denominator = 1;

            for (let j = 1; j < k; j++) {
                numerator -= phi[k-1][j] * acf[k-j];
                denominator -= phi[k-1][j] * acf[j];
            }

            phi[k][k] = numerator / denominator;
            pacf[k] = phi[k][k];

            // Update phi matrix
            for (let j = 1; j < k; j++) {
                phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k-j];
            }
        }

        // Calculate confidence intervals (±1.96/√n)
        const ci = 1.96 / Math.sqrt(data.length);

        return {
            pacf,
            upperCI: Array(maxLag + 1).fill(ci),
            lowerCI: Array(maxLag + 1).fill(-ci)
        };
    }

    // 7. Classification Methods
    knn(trainingX, trainingY, testX, k = 3, distanceMetric = 'euclidean') {
        if (!Array.isArray(trainingX) || !trainingX.every(Array.isArray)) {
            throw new Error('Training X must be a 2D array');
        }
        if (!Array.isArray(testX) || !testX.every(Array.isArray)) {
            throw new Error('Test X must be a 2D array');
        }
        if (trainingX.length !== trainingY.length) {
            throw new Error('Training X and Y must have the same length');
        }
        if (k < 1 || k > trainingX.length) {
            throw new Error('k must be between 1 and the number of training samples');
        }

        const predictions = [];
        
        for (const testPoint of testX) {
            // Calculate distances to all training points
            const distances = trainingX.map((trainPoint, index) => ({
                distance: this.#calculateDistance(testPoint, trainPoint, distanceMetric),
                label: trainingY[index]
            }));

            // Sort by distance and get k nearest neighbors
            distances.sort((a, b) => a.distance - b.distance);
            const kNearest = distances.slice(0, k);

            // Vote for the most common label
            const votes = {};
            kNearest.forEach(neighbor => {
                votes[neighbor.label] = (votes[neighbor.label] || 0) + 1;
            });

            // Find the label with the most votes
            let maxVotes = 0;
            let prediction = null;
            for (const label in votes) {
                if (votes[label] > maxVotes) {
                    maxVotes = votes[label];
                    prediction = label;
                }
            }

            predictions.push(prediction);
        }

        return predictions;
    }

    naiveBayes(trainingX, trainingY, testX) {
        if (!Array.isArray(trainingX) || !trainingX.every(Array.isArray)) {
            throw new Error('Training X must be a 2D array');
        }
        if (!Array.isArray(testX) || !testX.every(Array.isArray)) {
            throw new Error('Test X must be a 2D array');
        }
        if (trainingX.length !== trainingY.length) {
            throw new Error('Training X and Y must have the same length');
        }

        // Get unique classes
        const classes = [...new Set(trainingY)];
        const n_features = trainingX[0].length;

        // Calculate class priors and feature statistics
        const priors = {};
        const means = {};
        const variances = {};

        for (const c of classes) {
            // Get samples for this class
            const classIndices = trainingY.map((y, i) => y === c ? i : -1).filter(i => i !== -1);
            const classX = classIndices.map(i => trainingX[i]);

            // Calculate prior probability
            priors[c] = classIndices.length / trainingX.length;

            // Calculate mean and variance for each feature
            means[c] = Array(n_features).fill(0);
            variances[c] = Array(n_features).fill(0);

            // Calculate means
            for (let i = 0; i < n_features; i++) {
                means[c][i] = this.mean(classX.map(x => x[i]));
            }

            // Calculate variances
            for (let i = 0; i < n_features; i++) {
                variances[c][i] = this.variance(classX.map(x => x[i]));
            }
        }

        // Make predictions
        const predictions = testX.map(x => {
            const posteriors = {};

            for (const c of classes) {
                // Start with log prior to prevent underflow
                let logPosterior = Math.log(priors[c]);

                // Add log likelihoods for each feature
                for (let i = 0; i < n_features; i++) {
                    const mean = means[c][i];
                    const variance = variances[c][i];
                    
                    // Gaussian likelihood
                    logPosterior += this.#gaussianLogLikelihood(x[i], mean, variance);
                }

                posteriors[c] = logPosterior;
            }

            // Return class with highest posterior probability
            return Object.entries(posteriors).reduce((a, b) => 
                a[1] > b[1] ? a : b)[0];
        });

        return predictions;
    }

    decisionTree(trainingX, trainingY, testX, maxDepth = null) {
        if (!Array.isArray(trainingX) || !trainingX.every(Array.isArray)) {
            throw new Error('Training X must be a 2D array');
        }
        if (!Array.isArray(testX) || !testX.every(Array.isArray)) {
            throw new Error('Test X must be a 2D array');
        }
        if (trainingX.length !== trainingY.length) {
            throw new Error('Training X and Y must have the same length');
        }

        // Build the tree
        const tree = this.#buildTree(trainingX, trainingY, 0, maxDepth);

        // Make predictions
        return testX.map(x => this.#predictTree(tree, x));
    }

    // Helper methods for classification
    #calculateDistance(point1, point2, metric = 'euclidean') {
        if (point1.length !== point2.length) {
            throw new Error('Points must have the same dimensions');
        }

        switch (metric.toLowerCase()) {
            case 'euclidean':
                return Math.sqrt(point1.reduce((sum, x, i) => 
                    sum + Math.pow(x - point2[i], 2), 0));
            
            case 'manhattan':
                return point1.reduce((sum, x, i) => 
                    sum + Math.abs(x - point2[i]), 0);
            
            case 'cosine':
                const dot = point1.reduce((sum, x, i) => sum + x * point2[i], 0);
                const norm1 = Math.sqrt(point1.reduce((sum, x) => sum + x * x, 0));
                const norm2 = Math.sqrt(point2.reduce((sum, x) => sum + x * x, 0));
                return 1 - (dot / (norm1 * norm2));
            
            default:
                throw new Error('Unsupported distance metric');
        }
    }

    #gaussianLogLikelihood(x, mean, variance) {
        const eps = 1e-10; // Small constant to prevent division by zero
        return -0.5 * Math.log(2 * Math.PI * (variance + eps)) - 
               Math.pow(x - mean, 2) / (2 * (variance + eps));
    }

    #buildTree(X, y, depth = 0, maxDepth = null) {
        const n_samples = X.length;
        const n_features = X[0].length;

        // Check stopping criteria
        if (maxDepth !== null && depth >= maxDepth) {
            return this.#createLeaf(y);
        }

        const uniqueClasses = [...new Set(y)];
        if (uniqueClasses.length === 1) {
            return this.#createLeaf(y);
        }

        // Find best split
        let bestGini = Infinity;
        let bestFeature = null;
        let bestThreshold = null;
        let bestSplit = null;

        for (let feature = 0; feature < n_features; feature++) {
            const values = X.map(x => x[feature]);
            const uniqueValues = [...new Set(values)].sort((a, b) => a - b);

            // Try all possible thresholds
            for (let i = 0; i < uniqueValues.length - 1; i++) {
                const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
                const split = this.#splitData(X, y, feature, threshold);
                
                if (split.leftY.length > 0 && split.rightY.length > 0) {
                    const gini = this.#calculateGini(split);
                    if (gini < bestGini) {
                        bestGini = gini;
                        bestFeature = feature;
                        bestThreshold = threshold;
                        bestSplit = split;
                    }
                }
            }
        }

        // If no good split found, create leaf
        if (bestFeature === null) {
            return this.#createLeaf(y);
        }

        // Create node and recursively build subtrees
        return {
            feature: bestFeature,
            threshold: bestThreshold,
            left: this.#buildTree(bestSplit.leftX, bestSplit.leftY, depth + 1, maxDepth),
            right: this.#buildTree(bestSplit.rightX, bestSplit.rightY, depth + 1, maxDepth)
        };
    }

    #splitData(X, y, feature, threshold) {
        const leftIndices = [];
        const rightIndices = [];

        for (let i = 0; i < X.length; i++) {
            if (X[i][feature] <= threshold) {
                leftIndices.push(i);
            } else {
                rightIndices.push(i);
            }
        }

        return {
            leftX: leftIndices.map(i => X[i]),
            leftY: leftIndices.map(i => y[i]),
            rightX: rightIndices.map(i => X[i]),
            rightY: rightIndices.map(i => y[i])
        };
    }

    #calculateGini(split) {
        const leftGini = this.#giniImpurity(split.leftY);
        const rightGini = this.#giniImpurity(split.rightY);
        const totalSamples = split.leftY.length + split.rightY.length;

        return (split.leftY.length / totalSamples) * leftGini +
               (split.rightY.length / totalSamples) * rightGini;
    }

    #giniImpurity(y) {
        const counts = {};
        for (const label of y) {
            counts[label] = (counts[label] || 0) + 1;
        }

        const n = y.length;
        return 1 - Object.values(counts).reduce((sum, count) => 
            sum + Math.pow(count / n, 2), 0);
    }

    #createLeaf(y) {
        // Return most common class
        const counts = {};
        for (const label of y) {
            counts[label] = (counts[label] || 0) + 1;
        }
        return {
            value: Object.entries(counts).reduce((a, b) => 
                a[1] > b[1] ? a : b)[0]
        };
    }

    #predictTree(tree, x) {
        if ('value' in tree) {
            return tree.value;
        }

        if (x[tree.feature] <= tree.threshold) {
            return this.#predictTree(tree.left, x);
        } else {
            return this.#predictTree(tree.right, x);
        }
    }

    // 8. Clustering Methods
    kMeans(data, k, maxIterations = 100, tolerance = 1e-4) {
        if (!Array.isArray(data) || !data.every(Array.isArray)) {
            throw new Error('Data must be a 2D array');
        }
        if (k < 1 || k > data.length) {
            throw new Error('k must be between 1 and the number of data points');
        }

        const n_samples = data.length;
        const n_features = data[0].length;

        // Initialize centroids using k-means++
        const centroids = this.#initializeCentroids(data, k);
        let oldCentroids = Array(k).fill().map(() => Array(n_features).fill(Infinity));
        let labels = Array(n_samples).fill(0);
        let iteration = 0;

        while (iteration < maxIterations) {
            // Assign points to nearest centroid
            for (let i = 0; i < n_samples; i++) {
                let minDistance = Infinity;
                let minCluster = 0;

                for (let j = 0; j < k; j++) {
                    const distance = this.#calculateDistance(data[i], centroids[j]);
                    if (distance < minDistance) {
                        minDistance = distance;
                        minCluster = j;
                    }
                }

                labels[i] = minCluster;
            }

            // Update centroids
            for (let i = 0; i < k; i++) {
                const clusterPoints = data.filter((_, idx) => labels[idx] === i);
                if (clusterPoints.length > 0) {
                    for (let j = 0; j < n_features; j++) {
                        centroids[i][j] = this.mean(clusterPoints.map(p => p[j]));
                    }
                }
            }

            // Check convergence
            let centroidShift = 0;
            for (let i = 0; i < k; i++) {
                centroidShift += this.#calculateDistance(centroids[i], oldCentroids[i]);
            }

            if (centroidShift < tolerance) {
                break;
            }

            oldCentroids = centroids.map(c => [...c]);
            iteration++;
        }

        // Calculate inertia (within-cluster sum of squares)
        const inertia = data.reduce((sum, point, i) => 
            sum + Math.pow(this.#calculateDistance(point, centroids[labels[i]]), 2), 0);

        return {
            labels,
            centroids,
            inertia,
            iterations: iteration + 1
        };
    }

    dbscan(data, eps, minPts) {
        if (!Array.isArray(data) || !data.every(Array.isArray)) {
            throw new Error('Data must be a 2D array');
        }
        if (eps <= 0) {
            throw new Error('eps must be positive');
        }
        if (minPts < 1) {
            throw new Error('minPts must be at least 1');
        }

        const n_samples = data.length;
        const labels = Array(n_samples).fill(-1); // -1 represents unvisited points
        let currentCluster = 0;

        // Find neighbors for each point
        const neighbors = data.map((point1, i) => 
            data.reduce((acc, point2, j) => {
                if (i !== j && this.#calculateDistance(point1, point2) <= eps) {
                    acc.push(j);
                }
                return acc;
            }, [])
        );

        // Process each point
        for (let i = 0; i < n_samples; i++) {
            if (labels[i] !== -1) continue; // Skip if already processed

            const pointNeighbors = neighbors[i];
            if (pointNeighbors.length < minPts) {
                labels[i] = 0; // Mark as noise
                continue;
            }

            // Start new cluster
            currentCluster++;
            labels[i] = currentCluster;

            // Process neighbors
            const seedSet = [...pointNeighbors];
            for (let j = 0; j < seedSet.length; j++) {
                const currentPoint = seedSet[j];
                
                // Mark noise points as border points
                if (labels[currentPoint] === 0) {
                    labels[currentPoint] = currentCluster;
                }
                
                // Skip if already processed
                if (labels[currentPoint] !== -1) continue;

                // Add to cluster
                labels[currentPoint] = currentCluster;

                // Add neighbors to seed set if core point
                const currentNeighbors = neighbors[currentPoint];
                if (currentNeighbors.length >= minPts) {
                    seedSet.push(...currentNeighbors.filter(n => !seedSet.includes(n)));
                }
            }
        }

        return {
            labels,
            nClusters: currentCluster,
            noise: labels.filter(l => l === 0).length
        };
    }

    hierarchicalClustering(data, nClusters, linkage = 'complete') {
        if (!Array.isArray(data) || !data.every(Array.isArray)) {
            throw new Error('Data must be a 2D array');
        }
        if (nClusters < 1 || nClusters > data.length) {
            throw new Error('nClusters must be between 1 and the number of data points');
        }

        const n_samples = data.length;
        
        // Initialize clusters (each point in its own cluster)
        let clusters = data.map((point, i) => ({
            points: [i],
            centroid: [...point]
        }));

        // Initialize distance matrix
        let distances = Array(n_samples).fill().map(() => Array(n_samples).fill(Infinity));
        for (let i = 0; i < n_samples; i++) {
            for (let j = i + 1; j < n_samples; j++) {
                distances[i][j] = distances[j][i] = this.#calculateDistance(data[i], data[j]);
            }
        }

        // Merge clusters until desired number is reached
        while (clusters.length > nClusters) {
            // Find closest clusters
            let minDist = Infinity;
            let toMerge = [0, 1];

            for (let i = 0; i < clusters.length; i++) {
                for (let j = i + 1; j < clusters.length; j++) {
                    const dist = this.#calculateClusterDistance(
                        clusters[i], clusters[j], distances, linkage
                    );
                    if (dist < minDist) {
                        minDist = dist;
                        toMerge = [i, j];
                    }
                }
            }

            // Merge clusters
            const [i, j] = toMerge;
            const newCluster = {
                points: [...clusters[i].points, ...clusters[j].points],
                centroid: clusters[i].points.map((_, idx) => 
                    (clusters[i].centroid[idx] * clusters[i].points.length + 
                     clusters[j].centroid[idx] * clusters[j].points.length) / 
                    (clusters[i].points.length + clusters[j].points.length)
                )
            };

            // Update clusters array
            clusters = [
                ...clusters.slice(0, i),
                newCluster,
                ...clusters.slice(i + 1, j),
                ...clusters.slice(j + 1)
            ];
        }

        // Assign labels to points
        const labels = Array(n_samples).fill(0);
        clusters.forEach((cluster, i) => {
            cluster.points.forEach(point => {
                labels[point] = i;
            });
        });

        return {
            labels,
            nClusters: clusters.length,
            clusters: clusters.map(c => ({
                points: c.points,
                centroid: c.centroid
            }))
        };
    }

    // Helper methods for clustering
    #initializeCentroids(data, k) {
        const n_samples = data.length;
        const n_features = data[0].length;
        const centroids = Array(k).fill().map(() => Array(n_features));
        
        // Choose first centroid randomly
        const firstIdx = Math.floor(Math.random() * n_samples);
        centroids[0] = [...data[firstIdx]];

        // Choose remaining centroids
        for (let i = 1; i < k; i++) {
            // Calculate distances to nearest centroid for each point
            const distances = data.map(point => {
                let minDist = Infinity;
                for (let j = 0; j < i; j++) {
                    const dist = this.#calculateDistance(point, centroids[j]);
                    minDist = Math.min(minDist, dist);
                }
                return Math.pow(minDist, 2);
            });

            // Choose next centroid with probability proportional to distance
            const totalDist = distances.reduce((a, b) => a + b, 0);
            let rand = Math.random() * totalDist;
            let sum = 0;
            let nextIdx = 0;

            for (let j = 0; j < n_samples; j++) {
                sum += distances[j];
                if (sum > rand) {
                    nextIdx = j;
                    break;
                }
            }

            centroids[i] = [...data[nextIdx]];
        }

        return centroids;
    }

    #calculateClusterDistance(cluster1, cluster2, distances, linkage) {
        const dists = [];
        for (const p1 of cluster1.points) {
            for (const p2 of cluster2.points) {
                dists.push(distances[p1][p2]);
            }
        }

        switch (linkage) {
            case 'single':
                return Math.min(...dists);
            case 'complete':
                return Math.max(...dists);
            case 'average':
                return this.mean(dists);
            default:
                throw new Error('Unsupported linkage method');
        }
    }

    #calculateRanks(arr) {
        const sorted = arr.map((value, index) => ({ value, index }))
            .sort((a, b) => a.value - b.value);
        
        const ranks = new Array(arr.length);
        let currentRank = 1;
        
        for (let i = 0; i < sorted.length; i++) {
            const j = i;
            while (i < sorted.length - 1 && sorted[i].value === sorted[i + 1].value) {
                i++;
            }
            
            // Calculate average rank for ties
            const rank = (currentRank + (i - j) / 2);
            for (let k = j; k <= i; k++) {
                ranks[sorted[k].index] = rank;
            }
            
            currentRank = i + 2;
        }
        
        return ranks;
    }

    kendallCorrelation(x, y) {
        if (x.length !== y.length) {
            throw new Error('Arrays must have equal length');
        }
        this.#validateArray(x);
        this.#validateArray(y);

        const n = x.length;
        let concordant = 0;
        let discordant = 0;

        for (let i = 0; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) {
                const xDiff = x[i] - x[j];
                const yDiff = y[i] - y[j];
                if (xDiff * yDiff > 0) concordant++;
                else if (xDiff * yDiff < 0) discordant++;
            }
        }

        const tau = (concordant - discordant) / (0.5 * n * (n - 1));
        const z = 3 * tau * Math.sqrt(n * (n - 1)) / Math.sqrt(2 * (2 * n + 5));
        const pValue = 2 * (1 - this.#normalCDF(Math.abs(z)));

        return {
            coefficient: tau,
            z,
            pValue
        };
    }

    // 9. Dimensionality Reduction
    pca(data, nComponents = null) {
        if (!Array.isArray(data) || !data.every(Array.isArray)) {
            throw new Error('Data must be a 2D array');
        }

        const n_samples = data.length;
        const n_features = data[0].length;

        if (nComponents === null) {
            nComponents = n_features;
        } else if (nComponents < 1 || nComponents > n_features) {
            throw new Error('nComponents must be between 1 and number of features');
        }

        // Center the data
        const means = Array(n_features).fill(0);
        for (let j = 0; j < n_features; j++) {
            means[j] = this.mean(data.map(x => x[j]));
        }
        const centeredData = data.map(row => 
            row.map((val, j) => val - means[j])
        );

        // Calculate covariance matrix
        const covMatrix = Array(n_features).fill().map(() => Array(n_features).fill(0));
        for (let i = 0; i < n_features; i++) {
            for (let j = i; j < n_features; j++) {
                const cov = this.#covariance(
                    centeredData.map(x => x[i]),
                    centeredData.map(x => x[j])
                );
                covMatrix[i][j] = covMatrix[j][i] = cov;
            }
        }

        // Calculate eigenvalues and eigenvectors using power iteration
        const { eigenvalues, eigenvectors } = this.#powerIteration(covMatrix, nComponents);

        // Sort eigenvectors by eigenvalues in descending order
        const indices = eigenvalues.map((val, idx) => idx)
            .sort((a, b) => eigenvalues[b] - eigenvalues[a]);
        
        const sortedEigenvalues = indices.map(i => eigenvalues[i]);
        const sortedEigenvectors = indices.map(i => eigenvectors[i]);

        // Project data onto principal components
        const transformedData = centeredData.map(row => 
            sortedEigenvectors.slice(0, nComponents).map(vec => 
                vec.reduce((sum, v, j) => sum + v * row[j], 0)
            )
        );

        // Calculate explained variance ratio
        const totalVariance = sortedEigenvalues.reduce((a, b) => a + b, 0);
        const explainedVarianceRatio = sortedEigenvalues
            .slice(0, nComponents)
            .map(val => val / totalVariance);

        return {
            transformedData,
            components: sortedEigenvectors.slice(0, nComponents),
            explainedVarianceRatio,
            means
        };
    }

    tsne(data, nComponents = 2, perplexity = 30, iterations = 1000, learningRate = 200) {
        if (!Array.isArray(data) || !data.every(Array.isArray)) {
            throw new Error('Data must be a 2D array');
        }
        if (nComponents < 1 || nComponents > 3) {
            throw new Error('nComponents must be between 1 and 3');
        }

        const n_samples = data.length;
        const n_features = data[0].length;

        // Calculate pairwise distances
        const distances = Array(n_samples).fill().map(() => Array(n_samples).fill(0));
        for (let i = 0; i < n_samples; i++) {
            for (let j = i + 1; j < n_samples; j++) {
                distances[i][j] = distances[j][i] = 
                    this.#calculateDistance(data[i], data[j]);
            }
        }

        // Calculate joint probabilities (P)
        const P = this.#computeJointProbabilities(distances, perplexity);

        // Initialize low-dimensional representation
        const Y = Array(n_samples).fill().map(() => 
            Array(nComponents).fill().map(() => 
                Math.random() * 0.0001
            )
        );

        // Gradient descent
        for (let iter = 0; iter < iterations; iter++) {
            // Calculate low-dimensional pairwise affinities (Q)
            const Q = this.#computeLowDimAffinities(Y);

            // Calculate gradients
            const dY = Array(n_samples).fill().map(() => Array(nComponents).fill(0));
            for (let i = 0; i < n_samples; i++) {
                for (let j = 0; j < n_samples; j++) {
                    if (i === j) continue;
                    const force = 4 * (P[i][j] - Q[i][j]);
                    for (let k = 0; k < nComponents; k++) {
                        dY[i][k] += force * (Y[i][k] - Y[j][k]);
                    }
                }
            }

            // Update Y
            const currentLearningRate = learningRate / (1 + iter / iterations);
            for (let i = 0; i < n_samples; i++) {
                for (let k = 0; k < nComponents; k++) {
                    Y[i][k] -= currentLearningRate * dY[i][k];
                }
            }
        }

        return Y;
    }

    // Helper methods for dimensionality reduction
    #covariance(x, y) {
        const n = x.length;
        const meanX = this.mean(x);
        const meanY = this.mean(y);
        return x.reduce((sum, xi, i) => 
            sum + (xi - meanX) * (y[i] - meanY), 0) / (n - 1);
    }

    #powerIteration(matrix, nComponents) {
        const n = matrix.length;
        const eigenvalues = [];
        const eigenvectors = [];
        let remainingMatrix = matrix.map(row => [...row]);

        for (let k = 0; k < nComponents; k++) {
            // Initialize random vector
            let vector = Array(n).fill().map(() => Math.random());
            const norm = Math.sqrt(vector.reduce((sum, x) => sum + x * x, 0));
            vector = vector.map(x => x / norm);

            // Power iteration
            for (let iter = 0; iter < 100; iter++) {
                // Matrix-vector multiplication
                const newVector = Array(n).fill(0);
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        newVector[i] += remainingMatrix[i][j] * vector[j];
                    }
                }

                // Normalize
                const newNorm = Math.sqrt(newVector.reduce((sum, x) => sum + x * x, 0));
                vector = newVector.map(x => x / newNorm);
            }

            // Calculate eigenvalue
            let eigenvalue = 0;
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    eigenvalue += vector[i] * remainingMatrix[i][j] * vector[j];
                }
            }

            eigenvalues.push(eigenvalue);
            eigenvectors.push(vector);

            // Deflate matrix
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    remainingMatrix[i][j] -= eigenvalue * vector[i] * vector[j];
                }
            }
        }

        return { eigenvalues, eigenvectors };
    }

    #computeJointProbabilities(distances, perplexity) {
        const n = distances.length;
        const P = Array(n).fill().map(() => Array(n).fill(0));

        // Binary search for sigma
        for (let i = 0; i < n; i++) {
            let betaMin = -Infinity;
            let betaMax = Infinity;
            let beta = 1.0;
            let tries = 0;
            const target = Math.log(perplexity);

            while (tries < 50) {
                // Compute conditional probabilities
                let sum = 0;
                for (let j = 0; j < n; j++) {
                    if (i === j) continue;
                    P[i][j] = Math.exp(-distances[i][j] * beta);
                    sum += P[i][j];
                }

                if (sum === 0) sum = 1e-12;
                
                // Normalize and compute entropy
                let entropy = 0;
                for (let j = 0; j < n; j++) {
                    if (i === j) continue;
                    P[i][j] /= sum;
                    if (P[i][j] > 1e-12) {
                        entropy -= P[i][j] * Math.log(P[i][j]);
                    }
                }

                // Update beta based on error
                const error = entropy - target;
                if (error > 0) {
                    betaMin = beta;
                    beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2;
                } else {
                    betaMax = beta;
                    beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2;
                }
                tries++;
            }
        }

        // Symmetrize P
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                P[i][j] = (P[i][j] + P[j][i]) / (2 * n);
            }
        }

        return P;
    }

    #computeLowDimAffinities(Y) {
        const n = Y.length;
        const Q = Array(n).fill().map(() => Array(n).fill(0));
        let Z = 0;

        // Compute unnormalized Q
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const dist = this.#calculateDistance(Y[i], Y[j]);
                const q = 1 / (1 + dist * dist);
                Q[i][j] = Q[j][i] = q;
                Z += 2 * q;
            }
        }

        // Normalize Q
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                Q[i][j] /= Z;
            }
        }

        return Q;
    }

    // 10. Optimization Methods
    gradientDescent(objective, initialParams, learningRate = 0.01, maxIterations = 1000, tolerance = 1e-6) {
        if (!Array.isArray(initialParams)) {
            throw new Error('Initial parameters must be an array');
        }

        let params = [...initialParams];
        let prevLoss = Infinity;
        let iteration = 0;

        while (iteration < maxIterations) {
            // Calculate gradients using finite differences
            const gradients = params.map((param, i) => {
                const h = Math.max(Math.abs(param) * 1e-7, 1e-7);
                const paramsPlus = [...params];
                paramsPlus[i] += h;
                const paramsMin = [...params];
                paramsMin[i] -= h;
                return (objective(paramsPlus) - objective(paramsMin)) / (2 * h);
            });

            // Update parameters
            params = params.map((param, i) => param - learningRate * gradients[i]);

            // Calculate current loss
            const currentLoss = objective(params);

            // Check convergence
            if (Math.abs(currentLoss - prevLoss) < tolerance) {
                break;
            }

            prevLoss = currentLoss;
            iteration++;
        }

        return {
            params,
            loss: prevLoss,
            iterations: iteration + 1,
            converged: iteration < maxIterations
        };
    }

    newtonRaphson(objective, gradient, hessian, initialParams, maxIterations = 100, tolerance = 1e-6) {
        if (!Array.isArray(initialParams)) {
            throw new Error('Initial parameters must be an array');
        }

        let params = [...initialParams];
        let iteration = 0;

        while (iteration < maxIterations) {
            // Calculate gradient and Hessian
            const grad = gradient(params);
            const hess = hessian(params);

            // Solve Hessian * delta = -gradient
            const delta = this.#solveLinearSystem(hess, grad.map(x => -x));

            // Update parameters
            params = params.map((param, i) => param + delta[i]);

            // Check convergence
            if (Math.sqrt(grad.reduce((sum, x) => sum + x * x, 0)) < tolerance) {
                break;
            }

            iteration++;
        }

        return {
            params,
            loss: objective(params),
            iterations: iteration + 1,
            converged: iteration < maxIterations
        };
    }

    conjugateGradient(objective, gradient, initialParams, maxIterations = 1000, tolerance = 1e-6) {
        if (!Array.isArray(initialParams)) {
            throw new Error('Initial parameters must be an array');
        }

        let params = [...initialParams];
        let grad = gradient(params);
        let direction = grad.map(x => -x);
        let iteration = 0;

        while (iteration < maxIterations) {
            // Line search using backtracking
            let alpha = 1.0;
            const c = 0.5;
            const rho = 0.8;
            
            while (objective(params.map((p, i) => p + alpha * direction[i])) >
                   objective(params) + c * alpha * grad.reduce((sum, g, i) => sum + g * direction[i], 0)) {
                alpha *= rho;
            }

            // Update parameters
            params = params.map((p, i) => p + alpha * direction[i]);

            // Calculate new gradient
            const newGrad = gradient(params);

            // Check convergence
            if (Math.sqrt(newGrad.reduce((sum, x) => sum + x * x, 0)) < tolerance) {
                break;
            }

            // Calculate beta using Polak-Ribière formula
            const beta = Math.max(0,
                newGrad.reduce((sum, g, i) => sum + g * (newGrad[i] - grad[i]), 0) /
                grad.reduce((sum, g) => sum + g * g, 0)
            );

            // Update direction
            direction = newGrad.map((g, i) => -g + beta * direction[i]);
            grad = newGrad;

            iteration++;
        }

        return {
            params,
            loss: objective(params),
            iterations: iteration + 1,
            converged: iteration < maxIterations
        };
    }

    // Helper method for optimization
    #solveLinearSystem(A, b) {
        const n = A.length;
        const x = new Array(n).fill(0);
        const L = Array(n).fill().map(() => Array(n).fill(0));
        const U = Array(n).fill().map(() => Array(n).fill(0));

        // LU decomposition
        for (let i = 0; i < n; i++) {
            // Upper triangular
            for (let k = i; k < n; k++) {
                let sum = 0;
                for (let j = 0; j < i; j++) {
                    sum += L[i][j] * U[j][k];
                }
                U[i][k] = A[i][k] - sum;
            }

            // Lower triangular
            for (let k = i; k < n; k++) {
                if (i === k) {
                    L[i][i] = 1;
                } else {
                    let sum = 0;
                    for (let j = 0; j < i; j++) {
                        sum += L[k][j] * U[j][i];
                    }
                    L[k][i] = (A[k][i] - sum) / U[i][i];
                }
            }
        }

        // Forward substitution (Ly = b)
        const y = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = 0; j < i; j++) {
                sum += L[i][j] * y[j];
            }
            y[i] = b[i] - sum;
        }

        // Backward substitution (Ux = y)
        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) {
                sum += U[i][j] * x[j];
            }
            x[i] = (y[i] - sum) / U[i][i];
        }

        return x;
    }

    // 11. Neural Networks
    NeuralNetwork = class {
        constructor(layers) {
            this.layers = layers;
            this.weights = [];
            this.biases = [];
            
            // Initialize weights and biases
            for (let i = 0; i < layers.length - 1; i++) {
                const w = Array(layers[i + 1]).fill().map(() => 
                    Array(layers[i]).fill().map(() => 
                        Math.random() * 2 - 1
                    )
                );
                const b = Array(layers[i + 1]).fill().map(() => 
                    Math.random() * 2 - 1
                );
                this.weights.push(w);
                this.biases.push(b);
            }
        }

        forward(input) {
            let activation = input;
            const activations = [input];
            const zs = [];

            // Forward propagation
            for (let i = 0; i < this.weights.length; i++) {
                const z = this.#add(
                    this.#matmul(this.weights[i], activation),
                    this.biases[i]
                );
                zs.push(z);
                activation = this.#sigmoid(z);
                activations.push(activation);
            }

            return {
                output: activation,
                activations,
                zs
            };
        }

        backward(x, y) {
            const nabla_w = this.weights.map(w => 
                w.map(row => Array(row.length).fill(0))
            );
            const nabla_b = this.biases.map(b => 
                Array(b.length).fill(0)
            );

            // Forward pass
            const { output, activations, zs } = this.forward(x);

            // Backward pass
            let delta = this.#costDerivative(output, y);
            delta = delta.map((d, i) => 
                d * this.#sigmoidPrime(zs[zs.length - 1][i])
            );

            nabla_b[nabla_b.length - 1] = delta;
            nabla_w[nabla_w.length - 1] = this.#outer(delta, activations[activations.length - 2]);

            for (let l = 2; l < this.layers.length; l++) {
                const z = zs[zs.length - l];
                const sp = this.#sigmoidPrime(z);
                delta = this.#multiply(
                    this.#matmul(
                        this.#transpose(this.weights[this.weights.length - l + 1]),
                        delta
                    ),
                    sp
                );
                nabla_b[nabla_b.length - l] = delta;
                nabla_w[nabla_w.length - l] = this.#outer(delta, activations[activations.length - l - 1]);
            }

            return { nabla_w, nabla_b };
        }

        train(trainingData, epochs, miniBatchSize, learningRate, testData = null) {
            const n = trainingData.length;
            const history = {
                trainLoss: [],
                testLoss: [],
                trainAccuracy: [],
                testAccuracy: []
            };

            for (let epoch = 0; epoch < epochs; epoch++) {
                // Shuffle training data
                trainingData = this.#shuffle(trainingData);

                // Create mini-batches
                const miniBatches = [];
                for (let k = 0; k < n; k += miniBatchSize) {
                    miniBatches.push(trainingData.slice(k, k + miniBatchSize));
                }

                // Update each mini-batch
                for (const miniBatch of miniBatches) {
                    this.#updateMiniBatch(miniBatch, learningRate);
                }

                // Calculate metrics
                const trainLoss = this.#evaluateLoss(trainingData);
                const trainAccuracy = this.#evaluateAccuracy(trainingData);
                history.trainLoss.push(trainLoss);
                history.trainAccuracy.push(trainAccuracy);

                if (testData) {
                    const testLoss = this.#evaluateLoss(testData);
                    const testAccuracy = this.#evaluateAccuracy(testData);
                    history.testLoss.push(testLoss);
                    history.testAccuracy.push(testAccuracy);
                }
            }

            return history;
        }

        predict(x) {
            return this.forward(x).output;
        }

        // Private helper methods
        #updateMiniBatch(miniBatch, learningRate) {
            const nabla_w = this.weights.map(w => 
                w.map(row => Array(row.length).fill(0))
            );
            const nabla_b = this.biases.map(b => 
                Array(b.length).fill(0)
            );

            for (const [x, y] of miniBatch) {
                const { nabla_w: dnw, nabla_b: dnb } = this.backward(x, y);
                nabla_w.forEach((nw, i) => {
                    nw.forEach((row, j) => {
                        row.forEach((_, k) => {
                            nabla_w[i][j][k] += dnw[i][j][k];
                        });
                    });
                });
                nabla_b.forEach((nb, i) => {
                    nb.forEach((_, j) => {
                        nabla_b[i][j] += dnb[i][j];
                    });
                });
            }

            // Update weights and biases
            const eta = learningRate / miniBatch.length;
            this.weights = this.weights.map((w, i) => 
                w.map((row, j) => 
                    row.map((val, k) => 
                        val - eta * nabla_w[i][j][k]
                    )
                )
            );
            this.biases = this.biases.map((b, i) => 
                b.map((val, j) => 
                    val - eta * nabla_b[i][j]
                )
            );
        }

        #evaluateLoss(data) {
            let loss = 0;
            for (const [x, y] of data) {
                const output = this.predict(x);
                loss += this.#mse(output, y);
            }
            return loss / data.length;
        }

        #evaluateAccuracy(data) {
            let correct = 0;
            for (const [x, y] of data) {
                const output = this.predict(x);
                if (this.#argmax(output) === this.#argmax(y)) {
                    correct++;
                }
            }
            return correct / data.length;
        }

        #sigmoid(z) {
            return Array.isArray(z) ?
                z.map(x => 1.0 / (1.0 + Math.exp(-x))) :
                1.0 / (1.0 + Math.exp(-z));
        }

        #sigmoidPrime(z) {
            const s = this.#sigmoid(z);
            return Array.isArray(z) ?
                s.map((x, i) => x * (1 - x)) :
                s * (1 - s);
        }

        #costDerivative(output, y) {
            return output.map((o, i) => o - y[i]);
        }

        #mse(output, y) {
            return output.reduce((sum, o, i) => 
                sum + Math.pow(o - y[i], 2), 0) / output.length;
        }

        #argmax(array) {
            return array.indexOf(Math.max(...array));
        }

        #matmul(a, b) {
            if (!Array.isArray(a[0])) a = [a];
            if (!Array.isArray(b[0])) b = b.map(x => [x]);
            
            const result = Array(a.length).fill().map(() => 
                Array(b[0].length).fill(0)
            );

            for (let i = 0; i < a.length; i++) {
                for (let j = 0; j < b[0].length; j++) {
                    for (let k = 0; k < b.length; k++) {
                        result[i][j] += a[i][k] * b[k][j];
                    }
                }
            }

            return result.length === 1 ? result[0] : result;
        }

        #transpose(matrix) {
            if (!Array.isArray(matrix[0])) return [matrix];
            return matrix[0].map((_, i) => 
                matrix.map(row => row[i])
            );
        }

        #outer(a, b) {
            return a.map(x => 
                b.map(y => x * y)
            );
        }

        #add(a, b) {
            return a.map((x, i) => x + b[i]);
        }

        #multiply(a, b) {
            return a.map((x, i) => x * b[i]);
        }

        #shuffle(array) {
            const result = [...array];
            for (let i = result.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [result[i], result[j]] = [result[j], result[i]];
            }
            return result;
        }
    }

    createNeuralNetwork(layers) {
        return new this.NeuralNetwork(layers);
    }
}
