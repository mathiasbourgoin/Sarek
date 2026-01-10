document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.code-tabs').forEach(tabContainer => {
    const headers = tabContainer.querySelectorAll('.tab-header');
    const contents = tabContainer.querySelectorAll('.tab-content');

    headers.forEach((header, index) => {
      header.addEventListener('click', () => {
        // Remove active class from all headers and contents
        headers.forEach(h => h.classList.remove('active'));
        contents.forEach(c => c.classList.remove('active'));

        // Add active class to current header and corresponding content
        header.classList.add('active');
        if (contents[index]) {
          contents[index].classList.add('active');
        }
      });
    });
  });
});