"use strict";
const $ = (id) => document.getElementById(id);
let selected = "",
  busy = false,
  uploadBusy = false,
  collections = [];
const showNotice = (text) => {
  $("notice").textContent = text;
  $("notice").hidden = !text;
};
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Sprawdź poprawność danych i spróbuj ponownie.",
    );
    error.status = response.status;
    throw error;
  }
  return data;
}
function setBusy(value) {
  busy = value;
  for (const id of [
    "send",
    "question",
    "rerank",
    "think",
    "add-collection",
    "refresh",
    "refresh-models",
  ])
    $(id).disabled =
      value || (id === "send" && !selected);
  $("model-select").disabled = value || !$("model-select").value;
  $("delete-collection").disabled = value || !selected;
  document
    .querySelectorAll(".collection, .suggestions button, .repeat-search")
    .forEach((button) => {
      button.disabled = value;
    });
  document.querySelectorAll(".generate-answer").forEach((button) => {
    button.disabled = value || !$("model-select").value || button.dataset.expired === "true";
  });
  document.querySelectorAll(".generation-hint").forEach((hint) => {
    hint.textContent = $("model-select").value
      ? `${$("model-select").value} · Think ${$("think").checked ? "włączony" : "wyłączony"}`
      : "Wybierz dostępny model, aby wygenerować odpowiedź.";
  });
}
function renderCollections() {
  $("collections").replaceChildren();
  if (!collections.length) {
    const note = document.createElement("p");
    note.className = "sidebar-note";
    note.textContent = "Jeszcze tu pusto. Dodaj pierwsze dokumenty.";
    $("collections").append(note);
  }
  for (const name of collections) {
    const button = document.createElement("button");
    button.className = "collection" + (name === selected ? " active" : "");
    button.disabled = busy;
    button.setAttribute("aria-pressed", String(name === selected));
    const icon = document.createElement("span");
    icon.className = "book-icon";
    icon.textContent = "▤";
    icon.setAttribute("aria-hidden", "true");
    const label = document.createElement("span");
    label.textContent = name;
    button.append(icon, label);
    button.onclick = () => selectCollection(name);
    $("collections").append(button);
  }
  $("collection-title").textContent = selected || "Twoja biblioteka";
  $("welcome-copy").textContent = selected
    ? "Najpierw znajdź fragmenty dokumentów. Jeśli potrzebujesz podsumowania, poproś model o odpowiedź."
    : "Dodaj dokumenty, żeby stworzyć swoją pierwszą kolekcję i rozpocząć rozmowę.";
  setBusy(busy);
}
function selectCollection(name) {
  if (selected !== name) {
    $("messages").replaceChildren();
    $("empty").hidden = false;
  }
  selected = name;
  showNotice("");
  renderCollections();
}
async function refresh(preferred) {
  const data = await api("/api/state");
  collections = data.collections;
  selectCollection(
    collections.includes(preferred ?? selected)
      ? (preferred ?? selected)
      : collections[0] || "",
  );
}
async function refreshModels() {
  const select = $("model-select");
  const current = select.value;
  $("refresh-models").disabled = true;
  try {
    const data = await api("/api/models");
    let saved = "";
    try {
      saved = localStorage.getItem("local-rag-model-fast-v1") || "";
    } catch {}
    const preferred = [current, saved, data.default].find((name) =>
      data.models.includes(name),
    );
    select.replaceChildren();
    for (const name of data.models) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name + (name === data.default ? " · zalecany" : "");
      select.append(option);
    }
    if (!data.models.length) {
      select.append(new Option("Brak modeli do rozmowy", ""));
      showNotice("Pobierz model do rozmowy w Ollamie i odśwież listę.");
    } else {
      select.value = preferred || data.models[0];
    }
  } catch (error) {
    select.replaceChildren(new Option("Ollama niedostępna", ""));
    showNotice(error.message);
  } finally {
    setBusy(busy);
  }
}
$("model-select").onchange = () => {
  try {
    localStorage.setItem("local-rag-model-fast-v1", $("model-select").value);
  } catch {}
  setBusy(busy);
};
$("refresh-models").onclick = refreshModels;
try {
  $("think").checked = localStorage.getItem("local-rag-think") === "true";
} catch {}
$("think").onchange = () => {
  try { localStorage.setItem("local-rag-think", String($("think").checked)); } catch {}
  setBusy(busy);
};

function message(text, role) {
  $("empty").hidden = true;
  const block = document.createElement("article");
  block.className = "message " + role;
  if (role === "user") block.textContent = text;
  else {
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = "LOCAL RAG";
    block.append(label);
  }
  $("messages").append(block);
  return block;
}
function renderAnswer(block, answer, seconds, model) {
  for (const part of answer.split("\n\n---\n\n")) {
    const [text, reference] = part.split("\n\nUzasadnienie: ");
    const p = document.createElement("p");
    p.className = "answer-text";
    p.textContent = text;
    block.append(p);
    if (reference) {
      const [quote, source] = reference.split("\n\nŹródło: ");
      const details = document.createElement("details"),
        summary = document.createElement("summary"),
        citation = document.createElement("blockquote");
      summary.textContent = "Źródło · " + (source || "Dokument");
      citation.textContent = quote;
      details.append(summary, citation);
      block.append(details);
    }
  }
  const time = document.createElement("span");
  time.className = "response-time";
  time.textContent = `${seconds.toFixed(1)} s · ${model} · analiza zakończona`;
  block.append(time);
}
function pendingStatus(response, initial) {
  const status = document.createElement("div");
  status.className = "pending";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");
  const text = document.createElement("span");
  text.textContent = initial + "…";
  status.append(spinner, text);
  response.append(status);
  const start = Date.now();
  let phase = initial;
  let polling = false;
  let stopped = false;
  const phases = {
    retrieval: "Wyszukuję źródła",
    reranking: "Wybieram najtrafniejsze fragmenty",
    generation: "Przygotowuję odpowiedź",
    verification: "Sprawdzam odpowiedź ze źródłami",
    done: "Odpowiedź gotowa",
    error: "Analiza przerwana",
  };
  const progressTimer = setInterval(async () => {
    if (polling) return;
    polling = true;
    try {
      const progress = await api("/api/progress");
      if (!stopped) phase = phases[progress.phase] || phase;
    } catch {
    } finally {
      polling = false;
    }
  }, 1500);
  const timer = setInterval(() => {
    text.textContent = `${phase}… ${Math.floor((Date.now() - start) / 1000)} s`;
  }, 1000);
  return () => {
    stopped = true;
    clearInterval(timer);
    clearInterval(progressTimer);
    status.remove();
  };
}

function renderSources(response, result) {
  response.querySelector(".message-label").textContent = "ZNALEZIONE FRAGMENTY";
  const info = document.createElement("p");
  info.className = "response-time";
  info.textContent = `Fragmenty: ${result.sources.length} · wyszukiwanie ${result.seconds.toFixed(1)} s`;
  response.append(info);
  if (!result.sources.length) {
    const empty = document.createElement("p");
    empty.textContent = "Nie znaleziono fragmentów. Spróbuj inaczej sformułować pytanie.";
    response.append(empty);
    return;
  }
  const actions = document.createElement("div");
  actions.className = "source-actions";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "primary generate-answer";
  button.textContent = "Wygeneruj odpowiedź";
  const hint = document.createElement("span");
  hint.className = "generation-hint";
  actions.append(button, hint);
  response.append(actions);
  const grid = document.createElement("div");
  grid.className = "source-grid";
  for (const source of result.sources) {
    const card = document.createElement("article");
    card.className = "source-card";
    const title = document.createElement("h3");
    title.textContent = `${source.id} · ${source.label}`;
    const preview = document.createElement("p");
    preview.className = "source-preview";
    const compact = source.text.replace(/\s+/g, " ").trim();
    preview.textContent = compact.length > 340
      ? compact.slice(0, 340).replace(/\s+\S*$/, "") + "…"
      : compact;
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Cały fragment";
    const quote = document.createElement("blockquote");
    quote.textContent = source.text;
    details.append(summary, quote);
    details.ontoggle = () => { preview.hidden = details.open; };
    card.append(title, preview, details);
    grid.append(card);
  }
  const output = document.createElement("section");
  output.className = "generated-answer";
  output.setAttribute("aria-label", "Odpowiedź modelu");
  response.append(grid, output);
  button.onclick = async () => {
    if (busy || !$("model-select").value) return;
    const model = $("model-select").value;
    const think = $("think").checked;
    setBusy(true);
    showNotice("");
    output.replaceChildren();
    const stop = pendingStatus(output, "Przygotowuję odpowiedź");
    output.scrollIntoView({ behavior: "smooth", block: "center" });
    try {
      const answer = await api("/api/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ search_id: result.search_id, model, think }),
      });
      stop();
      const title = document.createElement("h3");
      title.textContent = "Odpowiedź modelu";
      output.append(title);
      renderAnswer(output, answer.answer, answer.seconds, answer.model);
      if (answer.context_trimmed) {
        const note = document.createElement("p");
        note.className = "response-time";
        note.textContent = "Model otrzymał skrócony kontekst. Użyte fragmenty znajdziesz w źródłach odpowiedzi.";
        output.append(note);
      }
      button.textContent = "Wygeneruj ponownie";
    } catch (error) {
      const p = document.createElement("p");
      p.className = "error-text";
      p.textContent = error.message;
      output.append(p);
      if (error.status === 410 || error.status === 404) {
        button.dataset.expired = "true";
        const again = document.createElement("button");
        again.type = "button";
        again.className = "secondary repeat-search";
        again.textContent = "Wyszukaj ponownie";
        again.onclick = () => {
          $("question").value = result.question;
          $("question-form").requestSubmit();
        };
        output.append(again);
      }
    } finally {
      stop();
      setBusy(false);
    }
  };
}

$("question-form").onsubmit = async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (busy || !question) return;
  if (!selected) {
    showNotice("Najpierw dodaj lub wybierz kolekcję.");
    return;
  }
  showNotice("");
  message(question, "user");
  $("question").value = "";
  setBusy(true);
  const response = message("", "assistant");
  const stop = pendingStatus(response, "Wyszukuję źródła");
  response.scrollIntoView({ behavior: "smooth", block: "center" });
  try {
    const result = await api("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collection: selected, question, rerank: $("rerank").checked }),
    });
    stop();
    renderSources(response, result);
    response.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const p = document.createElement("p");
    p.className = "error-text";
    p.textContent = error.message;
    response.append(p);
    $("question").value = question;
  } finally {
    stop();
    setBusy(false);
    $("question").focus({ preventScroll: true });
  }
};
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("question-form").requestSubmit();
  }
});
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.onclick = () => {
    $("question").value = button.dataset.prompt;
    $("question").focus();
  };
});
$("refresh").onclick = () =>
  refresh().catch((error) => showNotice(error.message));
$("add-collection").onclick = () => {
  $("upload-status").textContent = "";
  $("upload-dialog").showModal();
};
document.querySelectorAll(".close-dialog").forEach((button) => {
  button.onclick = () => {
    if (!uploadBusy) button.closest("dialog").close();
  };
});
$("upload-dialog").addEventListener("cancel", (event) => {
  if (uploadBusy) event.preventDefault();
});
$("files").onchange = () => {
  $("file-summary").textContent =
    Array.from($("files").files)
      .map((file) => file.name)
      .join(", ") || "Nie wybrano plików.";
};
$("upload-form").onsubmit = async (event) => {
  event.preventDefault();
  if (busy) return;
  const files = Array.from($("files").files);
  if (
    !files.length ||
    files.length > 30 ||
    files.reduce((sum, file) => sum + file.size, 0) > 100 * 1024 * 1024
  ) {
    $("upload-status").textContent =
      "Wybierz 1–30 plików o łącznym rozmiarze do 100 MB.";
    return;
  }
  const form = new FormData();
  form.append("name", $("collection-name").value.trim());
  form.append("enrichment", String($("enrichment").checked));
  files.forEach((file) => form.append("files", file));
  setBusy(true);
  uploadBusy = true;
  $("upload-submit").disabled = true;
  $("upload-status").textContent =
    "Importuję dokumenty i buduję indeks. To może chwilę potrwać…";
  try {
    const result = await api("/api/collections", {
      method: "POST",
      body: form,
    });
    await refresh(result.collection);
    $("upload-dialog").close();
    $("upload-form").reset();
    $("file-summary").textContent = "Nie wybrano plików.";
    showNotice("Kolekcja gotowa. Możesz zapytać o swoje dokumenty.");
  } catch (error) {
    $("upload-status").textContent = error.message;
  } finally {
    uploadBusy = false;
    $("upload-submit").disabled = false;
    setBusy(false);
  }
};
$("delete-collection").onclick = () => {
  $("delete-copy").textContent = `Kolekcja „${selected}” zostanie usunięta.`;
  $("delete-dialog").showModal();
};
$("confirm-delete").onclick = async () => {
  if (busy || !selected) return;
  const name = selected;
  setBusy(true);
  $("confirm-delete").disabled = true;
  try {
    await api("/api/collections/" + encodeURIComponent(name), {
      method: "DELETE",
    });
    $("delete-dialog").close();
    await refresh();
    showNotice("Kolekcja została usunięta.");
  } catch (error) {
    $("delete-dialog").close();
    showNotice(error.message);
  } finally {
    $("confirm-delete").disabled = false;
    setBusy(false);
  }
};
refresh().catch((error) => {
  $("collections").replaceChildren();
  showNotice("Nie udało się wczytać biblioteki: " + error.message);
});

refreshModels();
