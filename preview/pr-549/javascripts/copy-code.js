document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('pre').forEach((pre) => {
    // Create button container
    const button = document.createElement('button');
    button.className = 'copy-code-btn';
    button.type = 'button';
    button.innerText = 'Copy';

    // Click handler
    button.addEventListener('click', () => {
      const code = pre.querySelector('code');
      const text = code ? code.innerText : pre.innerText;

      navigator.clipboard.writeText(text).then(() => {
        button.innerText = 'Copied!';
        button.classList.add('copied');
        
        setTimeout(() => {
          button.innerText = 'Copy';
          button.classList.remove('copied');
        }, 2000);
      }).catch(err => {
        console.error('Failed to copy:', err);
        button.innerText = 'Error';
      });
    });

    // Make pre relative so we can absolute position the button
    pre.style.position = 'relative';
    pre.appendChild(button);
  });
});
