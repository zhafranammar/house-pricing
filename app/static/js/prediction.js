(function () {
  const init = window.PREDICTION_INIT;
  if (!init) return;

  const explainUrl = init.explainUrl;
  let activeTask = init.activeTab || "regression";

  function activePane() {
    return document.querySelector(".tab-pane.active");
  }

  function readFormFromPane(pane) {
    if (!pane) return { ...init.form };
    const fd = new FormData(pane.querySelector("form"));
    return {
      square_feet: fd.get("square_feet") || init.form.square_feet,
      bedrooms: fd.get("bedrooms") || init.form.bedrooms,
      bathrooms: fd.get("bathrooms") || init.form.bathrooms,
      year_built: fd.get("year_built") || init.form.year_built,
      neighborhood: fd.get("neighborhood") || init.form.neighborhood,
      manual_price: fd.get("manual_price") || init.form.manual_price || "",
    };
  }

  function syncFields(source) {
    const name = source.name;
    if (!name) return;
    document.querySelectorAll('.pred-form [name="' + name + '"]').forEach(function (el) {
      if (el === source) return;
      if (el.type === "checkbox") el.checked = source.checked;
      else el.value = source.value;
    });
  }

  document.querySelectorAll('#predTabs button[data-task]').forEach(function (btn) {
    btn.addEventListener("shown.bs.tab", function () {
      activeTask = btn.getAttribute("data-task");
    });
  });

  document.querySelectorAll(".pred-form").forEach(function (form) {
    form.addEventListener("input", function (e) {
      if (e.target && e.target.name) syncFields(e.target);
    });
  });
})();
