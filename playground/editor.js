import { Statik } from '../statik.js';

// Initialize Statik and make it globally available
const stats = new Statik();
window.stats = stats; // Make instance available globally

// Initialize CodeMirror
const editor = CodeMirror.fromTextArea(document.getElementById("editor"), {
    mode: "javascript",
    theme: "github",
    lineNumbers: true,
    autoCloseBrackets: true,
    matchBrackets: true,
    indentUnit: 2,
    tabSize: 2,
    lineWrapping: true,
    viewportMargin: Infinity
});

window.csvData = null; // Store CSV data globally

// Custom console implementation
const output = document.getElementById('output');
const originalConsole = { ...console };

// Clear console and chart before each run
function clearOutput() {
    output.textContent = '';
    const chart = Chart.getChart('chart');
    if (chart) {
        chart.destroy();
    }
}

// Custom console implementation
const customConsole = {
    log: function(...args) {
        const formattedArgs = args.map(arg => {
            if (typeof arg === 'object' && arg !== null) {
                return JSON.stringify(arg, null, 2);
            }
            return String(arg);
        }).join(' ');
        output.textContent += formattedArgs + '\n';
        originalConsole.log(...args);
    },
    error: function(...args) {
        const formattedArgs = args.map(arg => {
            if (arg instanceof Error) {
                return arg.message;
            }
            return String(arg);
        }).join(' ');
        output.textContent += '🚫 Error: ' + formattedArgs + '\n';
        originalConsole.error(...args);
    }
};

// Handle CSV file upload
const csvFile = document.getElementById('csvFile');
const fileName = document.getElementById('fileName');

csvFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    fileName.textContent = file.name;
    
    try {
        const text = await file.text();
        window.csvData = parseCSV(text);
        
        // Auto-run analysis on the data
        const analysisCode = generateAnalysisCode(window.csvData);
        editor.setValue(analysisCode);
        
        // Auto-execute the code with a slight delay to ensure UI updates
        setTimeout(() => {
            executeCode();
        }, 100);
    } catch (error) {
        console.error('Error reading CSV:', error);
        fileName.textContent = 'Error reading file';
    }
});

function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = {};
    
    // Initialize arrays for each column
    headers.forEach(header => {
        data[header] = [];
    });
    
    // Parse data rows
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        headers.forEach((header, index) => {
            const value = values[index]?.trim();
            // Convert to number if possible
            const num = Number(value);
            data[header].push(isNaN(num) ? value : num);
        });
    }
    
    return data;
}

function generateAnalysisCode(data) {
    const numericColumns = Object.entries(data)
        .filter(([_, values]) => values.every(v => typeof v === 'number'))
        .map(([name]) => name);

    if (numericColumns.length === 0) {
        return '// No numeric columns found in the CSV data';
    }

    const column = numericColumns[0];
    const values = data[column];

    return `// Analyze ${column} from CSV data
const values = csvData['${column}'];

// Basic statistics
console.log('Basic Statistics for ${column}:');
console.log('Count:', values.length);
console.log('Mean:', stats.mean(values));
console.log('Median:', stats.median(values));
console.log('Mode:', stats.mode(values));
console.log('Standard Deviation:', stats.stddev(values));
console.log('Variance:', stats.variance(values));

// Advanced statistics
console.log('\\nAdvanced Statistics:');
console.log('Quartiles:', stats.quartiles(values));
console.log('IQR:', stats.iqr(values));
console.log('Skewness:', stats.skewness(values));
console.log('Kurtosis:', stats.kurtosis(values));

// Create a histogram
stats.visualize('histogram', {
    values: values,
    label: '${column} Distribution'
}, 'chart');`;
}

// Load example code into the editor
function loadExample(exampleId) {
    if (examples[exampleId]) {
        editor.setValue(examples[exampleId]);
        editor.refresh();
        clearOutput();
    }
}

// Handle tabs
const tabs = document.querySelectorAll('.tab-button');
const tabPanes = document.querySelectorAll('.tab-pane');

function switchTab(targetTab) {
    tabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === targetTab);
    });
    tabPanes.forEach(pane => {
        pane.classList.toggle('active', pane.id === `${targetTab}-tab`);
    });
}

tabs.forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// Execute code with proper console overriding
function executeCode() {
    clearOutput();
    const code = editor.getValue();

    // Override console methods
    const prevConsole = { ...console };
    Object.assign(console, customConsole);

    try {
        // Create a function from the editor content
        const fn = new Function(code);
        fn.call(window);

        // Switch to appropriate tab based on output
        setTimeout(() => {
            if (output.textContent.trim()) {
                switchTab('console');
            } else {
                const chartContext = document.querySelector('#chart').getContext('2d');
                const imageData = chartContext.getImageData(0, 0, 1, 1);
                if (imageData.data[3] > 0) {
                    switchTab('chart');
                }
            }
        }, 100);
    } catch (error) {
        console.error(error);
        switchTab('console');
    } finally {
        // Restore original console
        Object.assign(console, prevConsole);
    }
}

// Run button handler
document.getElementById('run').addEventListener('click', executeCode);

const examples = {
    // ... existing examples ...

    // Optimization Methods
    'optimization-gradient-descent': `// Example: Finding minimum of a quadratic function using gradient descent
const statik = new Statik();

// Define the objective function: f(x) = x^2 + 2x + 1
const objective = (x) => x[0] * x[0] + 2 * x[0] + 1;

// Initial parameters
const initialParams = [2.0];  // Start at x = 2
const learningRate = 0.1;
const maxIterations = 100;
const tolerance = 1e-6;

// Run gradient descent
const result = statik.gradientDescent(objective, initialParams, learningRate, maxIterations, tolerance);

console.log('Gradient Descent Results:');
console.log('Optimal parameters:', result.params);
console.log('Minimum value:', result.loss);
console.log('Iterations:', result.iterations);
console.log('Converged:', result.converged);`,

    'optimization-newton-raphson': `// Example: Finding root of a function using Newton-Raphson method
const statik = new Statik();

// Define the objective function: f(x) = x^3 - x - 2
const objective = (x) => x[0] * x[0] * x[0] - x[0] - 2;

// Define the gradient (derivative): f'(x) = 3x^2 - 1
const gradient = (x) => [3 * x[0] * x[0] - 1];

// Define the Hessian (second derivative): f''(x) = 6x
const hessian = (x) => [[6 * x[0]]];

// Initial guess
const initialParams = [2.0];
const maxIterations = 100;
const tolerance = 1e-6;

// Run Newton-Raphson method
const result = statik.newtonRaphson(objective, gradient, hessian, initialParams, maxIterations, tolerance);

console.log('Newton-Raphson Results:');
console.log('Root found at:', result.params);
console.log('Function value at root:', result.loss);
console.log('Iterations:', result.iterations);
console.log('Converged:', result.converged);`,

    'optimization-conjugate-gradient': `// Example: Minimizing Rosenbrock function using conjugate gradient
const statik = new Statik();

// Define the Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
const objective = (params) => {
    const x = params[0];
    const y = params[1];
    return Math.pow(1 - x, 2) + 100 * Math.pow(y - x * x, 2);
};

// Define the gradient
const gradient = (params) => {
    const x = params[0];
    const y = params[1];
    return [
        -2 * (1 - x) - 400 * x * (y - x * x),
        200 * (y - x * x)
    ];
};

// Initial guess
const initialParams = [0.0, 0.0];
const maxIterations = 1000;
const tolerance = 1e-6;

// Run conjugate gradient method
const result = statik.conjugateGradient(objective, gradient, initialParams, maxIterations, tolerance);

console.log('Conjugate Gradient Results:');
console.log('Optimal parameters:', result.params);
console.log('Minimum value:', result.loss);
console.log('Iterations:', result.iterations);
console.log('Converged:', result.converged);`,

    // Neural Networks
    'neural-network-xor': `// Example: Training a neural network on the XOR problem
const statik = new Statik();

// Create a neural network with 2 inputs, 4 hidden neurons, and 1 output
const nn = statik.createNeuralNetwork([2, 4, 1]);

// XOR training data: [input, expected_output]
const trainingData = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
];

// Training parameters
const epochs = 10000;
const miniBatchSize = 4;
const learningRate = 0.1;

// Train the network
const history = nn.train(trainingData, epochs, miniBatchSize, learningRate);

// Test the network
console.log('XOR Neural Network Results:');
trainingData.forEach(([input, expected]) => {
    const output = nn.predict(input);
    console.log(\`Input: \${input} | Expected: \${expected} | Output: \${output[0].toFixed(4)}\`);
});

console.log('\\nFinal training accuracy:', history.trainAccuracy[history.trainAccuracy.length - 1].toFixed(4));`,

    'neural-network-mnist': `// Example: Simple digit classification (simulated MNIST-like data)
const statik = new Statik();

// Create a neural network with 784 inputs (28x28), 128 hidden neurons, and 10 outputs
const nn = statik.createNeuralNetwork([784, 128, 10]);

// Generate some simple synthetic digit data (just for demonstration)
function generateDigitData(numSamples = 100) {
    const data = [];
    for (let i = 0; i < numSamples; i++) {
        // Create a simple pattern for digit 0-9
        const digit = Math.floor(Math.random() * 10);
        const input = new Array(784).fill(0);
        const output = new Array(10).fill(0);
        
        // Set some random pixels to 1 to simulate the digit
        for (let j = 0; j < 50; j++) {
            input[Math.floor(Math.random() * 784)] = 1;
        }
        
        output[digit] = 1;
        data.push([input, output]);
    }
    return data;
}

// Generate training and test data
const trainingData = generateDigitData(1000);
const testData = generateDigitData(100);

// Train the network
const history = nn.train(trainingData, 10, 32, 0.1, testData);

console.log('MNIST-like Neural Network Results:');
console.log('Final training accuracy:', history.trainAccuracy[history.trainAccuracy.length - 1].toFixed(4));
console.log('Final test accuracy:', history.testAccuracy[history.testAccuracy.length - 1].toFixed(4));

// Test some random digits
console.log('\\nPredictions on test data:');
for (let i = 0; i < 5; i++) {
    const [input, expected] = testData[i];
    const output = nn.predict(input);
    console.log(\`Expected: \${expected.indexOf(1)} | Predicted: \${output.indexOf(Math.max(...output))}\`);
}`,

    'neural-network-regression': `// Example: Function approximation using neural network
const statik = new Statik();

// Create a neural network with 1 input, 10 hidden neurons, and 1 output
const nn = statik.createNeuralNetwork([1, 10, 1]);

// Generate training data for y = sin(x)
function generateData(start, end, steps) {
    const data = [];
    const step = (end - start) / steps;
    for (let x = start; x <= end; x += step) {
        data.push([[x], [Math.sin(x)]]);
    }
    return data;
}

// Generate training and test data
const trainingData = generateData(-Math.PI, Math.PI, 100);
const testData = generateData(-Math.PI - 0.5, Math.PI + 0.5, 20);

// Train the network
const history = nn.train(trainingData, 1000, 32, 0.1, testData);

console.log('Function Approximation Results:');
console.log('Final training loss:', history.trainLoss[history.trainLoss.length - 1].toFixed(6));
console.log('Final test loss:', history.testLoss[history.testLoss.length - 1].toFixed(6));

// Test the network on some specific points
console.log('\\nPredictions:');
[-2, -1, 0, 1, 2].forEach(x => {
    const input = [x];
    const expected = Math.sin(x);
    const predicted = nn.predict(input)[0];
    console.log(\`x: \${x.toFixed(2)} | Expected: \${expected.toFixed(4)} | Predicted: \${predicted.toFixed(4)}\`);
});`
};
