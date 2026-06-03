window.transformMermaidBlocks = function() {
  // Common Jekyll/Kramdown selectors for mermaid code blocks
  const selectors = [
    'pre code.language-mermaid',
    'pre.mermaid code',
    'pre.mermaid',
    'code.language-mermaid'
  ];
  
  const foundBlocks = [];
  selectors.forEach(selector => {
    document.querySelectorAll(selector).forEach(block => {
      if (!foundBlocks.includes(block)) foundBlocks.push(block);
    });
  });

  foundBlocks.forEach(block => {
    let code = block.innerText;
    let target = block;
    
    // If it's a <code> inside a <pre>, we want to replace the <pre>
    if (block.tagName === 'CODE' && block.parentElement.tagName === 'PRE') {
      target = block.parentElement;
    }
    
    // Create the mermaid div
    const div = document.createElement('div');
    div.className = 'mermaid';
    div.textContent = code;
    
    // Replace the old block
    target.replaceWith(div);
  });
};