'use strict';
const AOIDrag = (() => {
  function inside(p, polygon) {
    let hit = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const a = polygon[i], b = polygon[j];
      if ((a[1] > p[1]) !== (b[1] > p[1]) &&
          p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]) hit = !hit;
    }
    return hit;
  }
  function translate(polygon, delta, size, screen) {
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const xs = (screen ? polygon.slice(1, -1) : polygon).map(p => p[0]);
    // Only screen x-order is constrained. Image borders clip drawing, not AOI geometry.
    const dx = screen ? clamp(delta[0], 3 - Math.min(...xs), size[0] - 4 - Math.max(...xs)) : delta[0];
    const dy = delta[1];
    return polygon.map((p, i) => [screen && i === 0 ? 0 : screen && i === polygon.length - 1 ? size[0] - 1 : p[0] + dx, p[1] + dy]);
  }
  return {inside, translate};
})();
if (typeof module !== 'undefined') module.exports = AOIDrag;
