
const statik = {
  mean(arr) {
    const sum = arr.reduce((a, b) => a + b, 0);
    return sum / arr.length;
  },
  median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  },
  variance(arr) {
    const m = statik.mean(arr);
    return statik.mean(arr.map(x => (x - m) ** 2));
  },
  stddev(arr) {
    return Math.sqrt(statik.variance(arr));
  },
  mode(arr) {
    const freq = {};
    arr.forEach(val => freq[val] = (freq[val] || 0) + 1);
    const maxFreq = Math.max(...Object.values(freq));
    return Object.entries(freq)
      .filter(([_, count]) => count === maxFreq)
      .map(([val]) => Number(val));
  }
};

export default statik;
