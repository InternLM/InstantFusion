/* Typeset code collage: decorative notation, not an executable method diagram. */
(() => {
  const fragments = [
    ['<latent>  shared', 'x[t]  /  E_A(x)', '     E_B(x)  /', '  z : common', '</latent>'],
    ['<flow>   native', '  x[t] − σ v[t]', 'd e n o i s e', '    → x[0]', '</flow>'],
    ['<interface>', ' E_A       E_B', '    ↘  z  ↙', ' D_A       D_B', '</interface>']
  ];
  document.querySelectorAll('.result-fragment, .rail-photo, .footer-graphic').forEach((host, index) => {
    const block = document.createElement('div');
    block.className = 'code-collage code-collage-' + index;
    block.setAttribute('aria-hidden', 'true');
    fragments[index].forEach((line, i) => {
      const span = document.createElement('span');
      span.textContent = line;
      span.style.setProperty('--line', i);
      block.append(span);
    });
    host.append(block);
  });
})();
