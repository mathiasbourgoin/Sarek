document.addEventListener('DOMContentLoaded', () => {
  const highlightSarek = () => {
    // Target code blocks
    const codeBlocks = document.querySelectorAll('pre code, .highlight pre, .tab-content pre');
    
    codeBlocks.forEach(block => {
      let html = block.innerHTML;
      
      // 1. Handle Sarek OCaml Extensions that might be split by Rouge
      // Regex: match % then optional tags, then the keyword. 
      // We wrap the whole thing.
      const extensions = ['kernel', 'shared', 'superstep'];
      extensions.forEach(ext => {
        const reg = new RegExp('%(?:<[^>]+>)*' + ext, 'g');
        html = html.replace(reg, (match) => `<span class="k">${match}</span>`);
      });

      // 2. Handle Sarek attributes
      const attributes = [/@sarek\.module/g, /@sarek\.type/g, /@@sarek\.type/g];
      attributes.forEach(attr => {
        html = html.replace(attr, (match) => `<span class="k">${match}</span>`);
      });

      // 3. Keywords (with fixed double-escaped word boundaries)
      const keywords = ['mut', '__kernel', '__global', '__local', 'get_global_id', 'get_local_id', 'barrier', 'threadgroup', 'thread_position_in_grid'];
      keywords.forEach(kw => {
        const reg = new RegExp('\\b' + kw + '\\b', 'g');
        if (!html.includes(`<span class="k">${kw}</span>`)) {
           html = html.replace(reg, `<span class="k">${kw}</span>`);
        }
      });

      block.innerHTML = html;
    });
  };

  highlightSarek();
  
  // Re-run on tab switch
  document.querySelectorAll('.tab-header').forEach(tab => {
    tab.addEventListener('click', () => {
      setTimeout(highlightSarek, 50);
    });
  });
});