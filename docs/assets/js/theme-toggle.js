(function () {
  "use strict";

  var storageKey = "theme-preference";
  var root = document.documentElement;
  var mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

  function savedTheme() {
    try {
      var value = window.localStorage.getItem(storageKey);
      return value === "light" || value === "dark" ? value : null;
    } catch (error) {
      return null;
    }
  }

  function preferredTheme() {
    return savedTheme() || (mediaQuery.matches ? "dark" : "light");
  }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    root.style.colorScheme = theme;

    var toggle = document.querySelector("[data-theme-toggle]");
    if (!toggle) return;

    var isDark = theme === "dark";
    toggle.setAttribute("aria-pressed", String(isDark));
    toggle.setAttribute(
      "aria-label",
      isDark ? "Switch to light mode" : "Switch to dark mode"
    );
    toggle.setAttribute(
      "title",
      isDark ? "Switch to light mode" : "Switch to dark mode"
    );
  }

  applyTheme(preferredTheme());

  document.addEventListener("DOMContentLoaded", function () {
    var navigation = document.querySelector("#site-nav") || document.querySelector(".greedy-nav");
    if (!navigation || navigation.querySelector("[data-theme-toggle]")) return;

    var toggle = document.createElement("button");
    toggle.className = "theme-toggle";
    toggle.type = "button";
    toggle.setAttribute("data-theme-toggle", "");
    toggle.innerHTML =
      '<svg class="theme-toggle__icon theme-toggle__icon--moon" aria-hidden="true" viewBox="0 0 24 24"><path d="M20.4 14.6A8.4 8.4 0 0 1 9.4 3.6 8.5 8.5 0 1 0 20.4 14.6Z"/></svg>' +
      '<svg class="theme-toggle__icon theme-toggle__icon--sun" aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.5"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42"/></svg>';

    var menuToggle = navigation.querySelector(".greedy-nav__toggle");
    navigation.insertBefore(toggle, menuToggle || null);
    applyTheme(preferredTheme());

    toggle.addEventListener("click", function () {
      var nextTheme = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      try {
        window.localStorage.setItem(storageKey, nextTheme);
      } catch (error) {
        // The toggle still works for this page when storage is unavailable.
      }
      applyTheme(nextTheme);
    });
  });

  function followSystemTheme(event) {
    if (!savedTheme()) applyTheme(event.matches ? "dark" : "light");
  }

  if (mediaQuery.addEventListener) {
    mediaQuery.addEventListener("change", followSystemTheme);
  } else if (mediaQuery.addListener) {
    mediaQuery.addListener(followSystemTheme);
  }
})();
