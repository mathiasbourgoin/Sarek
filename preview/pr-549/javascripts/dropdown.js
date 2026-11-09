// Accessible nav dropdown: click / keyboard / touch toggle.
// The CSS :hover rule keeps mouse behaviour; this adds the interactions hover
// can't provide (keyboard focus, touch tap, screen-reader state) and an Escape /
// outside-click close. Adding the `.open` class on .nav-dropdown reveals the menu
// (see modern.css), and aria-expanded is kept in sync for assistive tech.
(function () {
  function setupDropdown(dd) {
    var toggle = dd.querySelector('.dropdown-toggle');
    var menu = dd.querySelector('.dropdown-menu');
    if (!toggle || !menu) return;

    function close() {
      dd.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    }
    function open() {
      dd.classList.add('open');
      toggle.setAttribute('aria-expanded', 'true');
    }

    toggle.addEventListener('click', function (e) {
      e.preventDefault();
      if (dd.classList.contains('open')) close();
      else open();
    });

    // Escape closes and returns focus to the toggle.
    dd.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' || e.key === 'Esc') {
        close();
        toggle.focus();
      }
    });

    // Close when interaction leaves the dropdown.
    document.addEventListener('click', function (e) {
      if (!dd.contains(e.target)) close();
    });
    dd.addEventListener('focusout', function (e) {
      if (!dd.contains(e.relatedTarget)) close();
    });
  }

  document.querySelectorAll('.nav-dropdown').forEach(setupDropdown);
})();
