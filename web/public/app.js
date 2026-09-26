/**
 * LLM Streaming Chat Client Application
 * Connects to proxy server with real-time SSE token stream & Reasoning support
 */

// Application State
const state = {
  messages: [],
  activeModel: 'qwen3.8:27b',
  availableModels: [],
  isGenerating: false,
  abortController: null,
  systemPrompt: '',
  temperature: 0.7,
  maxTokens: 16384,
  thinkingMode: 'fast', // 'fast' (no-think) | 'deep' (reasoning)
  attachedImages: [], // Array of { id, dataUrl, name, size }
  serverStatus: 'checking', // 'online' | 'offline' | 'checking'
};

// DOM Elements
const elements = {
  chatContainer: document.getElementById('chatContainer'),
  messagesList: document.getElementById('messagesList'),
  welcomeScreen: document.getElementById('welcomeScreen'),
  promptInput: document.getElementById('promptInput'),
  sendBtn: document.getElementById('sendBtn'),
  attachBtn: document.getElementById('attachBtn'),
  imageFileInput: document.getElementById('imageFileInput'),
  imagePreviewBar: document.getElementById('imagePreviewBar'),
  inputContainer: document.getElementById('inputContainer'),
  inputSection: document.getElementById('inputSection'),
  imageLightboxModal: document.getElementById('imageLightboxModal'),
  lightboxOverlay: document.getElementById('lightboxOverlay'),
  lightboxImg: document.getElementById('lightboxImg'),
  lightboxCloseBtn: document.getElementById('lightboxCloseBtn'),
  modelSelector: document.getElementById('modelSelector'),
  modeBtns: document.querySelectorAll('.mode-btn'),
  thinkingModeSelect: document.getElementById('thinkingModeSelect'),
  promptPresetChips: document.querySelectorAll('.chip-btn[data-sys]'),
  sidebarStatusBadge: document.getElementById('sidebarStatusBadge'),
  topbarStatusPill: document.getElementById('topbarStatusPill'),
  activeModelDisplay: document.getElementById('activeModelDisplay'),
  endpointDisplay: document.getElementById('endpointDisplay'),
  newChatBtn: document.getElementById('newChatBtn'),
  clearChatBtn: document.getElementById('clearChatBtn'),
  themeToggleBtn: document.getElementById('themeToggleBtn'),
  sidebarToggleBtn: document.getElementById('sidebarToggleBtn'),
  sidebar: document.getElementById('sidebar'),
  settingsToggleBtn: document.getElementById('settingsToggleBtn'),
  settingsDrawer: document.getElementById('settingsDrawer'),
  closeSettingsBtn: document.getElementById('closeSettingsBtn'),
  systemPromptInput: document.getElementById('systemPromptInput'),
  tempInput: document.getElementById('tempInput'),
  tempVal: document.getElementById('tempVal'),
  maxTokensInput: document.getElementById('maxTokensInput'),
  maxTokensVal: document.getElementById('maxTokensVal'),
  streamStats: document.getElementById('streamStats'),
  tokenCountStat: document.getElementById('tokenCountStat'),
  speedStat: document.getElementById('speedStat'),
};

// Initialize Application
async function initApp() {
  setupEventListeners();
  loadSavedPreferences();
  await checkServerAndFetchModels();
}

/**
 * Fetch available models from backend /api/models
 */
async function checkServerAndFetchModels() {
  updateStatus('checking', '연결 확인 중...');
  try {
    const res = await fetch('/api/models');
    const data = await res.json();

    if (res.ok && data.success) {
      state.serverStatus = 'online';
      state.availableModels = data.models || [];
      
      // Update endpoint display if provided
      if (data.llmBaseUrl) {
        elements.endpointDisplay.textContent = data.llmBaseUrl.replace('http://', '');
      }

      // Populate selector
      elements.modelSelector.innerHTML = '';
      if (state.availableModels.length > 0) {
        state.availableModels.forEach((model) => {
          const opt = document.createElement('option');
          opt.value = model.id;
          opt.textContent = model.id;
          elements.modelSelector.appendChild(opt);
        });
        state.activeModel = state.availableModels[0].id;
      } else {
        const opt = document.createElement('option');
        opt.value = 'unsloth/Qwen3.8-27B-NVFP4';
        opt.textContent = 'unsloth/Qwen3.8-27B-NVFP4';
        elements.modelSelector.appendChild(opt);
      }

      elements.activeModelDisplay.textContent = state.activeModel;
      updateStatus('online', '온라인 (연결됨)');
    } else {
      throw new Error(data.error || 'Failed to fetch models');
    }
  } catch (err) {
    console.warn('Server check warning:', err.message);
    state.serverStatus = 'offline';
    updateStatus('offline', '오프라인 (확인 필요)');
    elements.activeModelDisplay.textContent = '연결 실패';
  }
}

/**
 * Update UI server status badges
 */
function updateStatus(status, text) {
  state.serverStatus = status;

  // Sidebar badge
  elements.sidebarStatusBadge.className = `status-badge ${status === 'online' ? '' : 'offline'}`;
  elements.sidebarStatusBadge.querySelector('.status-text').textContent = text;

  // Topbar pill
  elements.topbarStatusPill.className = `status-pill ${status === 'online' ? 'online' : 'offline'}`;
  elements.topbarStatusPill.querySelector('.status-label').textContent = status === 'online' ? 'LLM 연결됨' : 'LLM 연결 안됨';
}

/**
 * Setup Event Listeners
 */
function setupEventListeners() {
  // Textarea input resize & Enter key handling
  elements.promptInput.addEventListener('input', () => {
    elements.promptInput.style.height = 'auto';
    elements.promptInput.style.height = Math.min(elements.promptInput.scrollHeight, 180) + 'px';
  });

  elements.promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendPrompt();
    }
  });

  // Send / Stop button
  elements.sendBtn.addEventListener('click', () => {
    if (state.isGenerating) {
      stopGeneration();
    } else {
      handleSendPrompt();
    }
  });

  // New Chat / Clear
  elements.newChatBtn.addEventListener('click', startNewChat);
  elements.clearChatBtn.addEventListener('click', startNewChat);

  // Model Selection
  elements.modelSelector.addEventListener('change', (e) => {
    state.activeModel = e.target.value;
    elements.activeModelDisplay.textContent = state.activeModel;
  });

  // Preset prompts
  document.querySelectorAll('.preset-item, .feature-card').forEach((el) => {
    el.addEventListener('click', () => {
      const prompt = el.getAttribute('data-prompt');
      if (prompt) {
        elements.promptInput.value = prompt;
        elements.promptInput.style.height = 'auto';
        elements.promptInput.focus();
        handleSendPrompt();
      }
    });
  });

  // Theme Toggle
  elements.themeToggleBtn.addEventListener('click', () => {
    const isDark = document.body.classList.contains('dark-theme');
    document.body.classList.toggle('dark-theme', !isDark);
    document.body.classList.toggle('light-theme', isDark);
    localStorage.setItem('llm_theme', isDark ? 'light' : 'dark');
  });

  // Mobile Sidebar Toggle
  elements.sidebarToggleBtn.addEventListener('click', () => {
    elements.sidebar.classList.toggle('open');
  });

  // Settings Drawer Toggle
  elements.settingsToggleBtn.addEventListener('click', () => {
    elements.settingsDrawer.classList.toggle('open');
  });
  elements.closeSettingsBtn.addEventListener('click', () => {
    elements.settingsDrawer.classList.remove('open');
  });

  // Thinking Mode Switcher
  elements.modeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      setThinkingMode(btn.getAttribute('data-mode'));
    });
  });

  if (elements.thinkingModeSelect) {
    elements.thinkingModeSelect.addEventListener('change', (e) => {
      setThinkingMode(e.target.value);
    });
  }

  // System Prompt Presets
  elements.promptPresetChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const sysText = chip.getAttribute('data-sys');
      if (sysText) {
        elements.systemPromptInput.value = sysText;
        state.systemPrompt = sysText;
        // Visual feedback
        chip.style.borderColor = 'var(--success)';
        setTimeout(() => { chip.style.borderColor = ''; }, 600);
      }
    });
  });

  // Parameter Inputs
  elements.tempInput.addEventListener('input', (e) => {
    state.temperature = parseFloat(e.target.value);
    elements.tempVal.textContent = state.temperature.toFixed(2);
  });

  elements.maxTokensInput.addEventListener('input', (e) => {
    state.maxTokens = parseInt(e.target.value, 10);
    elements.maxTokensVal.textContent = state.maxTokens;
  });

  elements.systemPromptInput.addEventListener('input', (e) => {
    state.systemPrompt = e.target.value.trim();
  });

  // Vision: File Attachment Button & Input
  if (elements.attachBtn && elements.imageFileInput) {
    elements.attachBtn.addEventListener('click', () => {
      elements.imageFileInput.click();
    });

    elements.imageFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleImageFiles(e.target.files);
      }
    });
  }

  // Vision: Clipboard Paste (Ctrl+V / Cmd+V)
  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    const imgFiles = [];
    for (const item of items) {
      if (item.type && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) imgFiles.push(file);
      }
    }

    if (imgFiles.length > 0) {
      e.preventDefault();
      handleImageFiles(imgFiles);
    }
  });

  // Vision: Drag & Drop over Input Section
  if (elements.inputSection) {
    ['dragenter', 'dragover'].forEach(name => {
      elements.inputSection.addEventListener(name, (e) => {
        e.preventDefault();
        elements.inputSection.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      elements.inputSection.addEventListener(name, (e) => {
        e.preventDefault();
        elements.inputSection.classList.remove('drag-over');
      });
    });

    elements.inputSection.addEventListener('drop', (e) => {
      e.preventDefault();
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        const imgFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
        if (imgFiles.length > 0) {
          handleImageFiles(imgFiles);
        }
      }
    });
  }

  // Vision: Lightbox Modal Close Events
  if (elements.lightboxCloseBtn && elements.lightboxOverlay) {
    elements.lightboxCloseBtn.addEventListener('click', closeLightbox);
    elements.lightboxOverlay.addEventListener('click', closeLightbox);
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && elements.imageLightboxModal.style.display !== 'none') {
        closeLightbox();
      }
    });
  }
}

/**
 * Handle image files (Optimization & Base64 conversion)
 */
async function handleImageFiles(fileList) {
  for (const file of fileList) {
    if (!file.type.startsWith('image/')) continue;
    try {
      const dataUrl = await readAndOptimizeImage(file);
      state.attachedImages.push({
        id: 'img_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6),
        dataUrl,
        name: file.name,
        size: file.size
      });
    } catch (err) {
      console.warn('Failed to read image:', err);
    }
  }
  renderAttachedImages();
  elements.promptInput.focus();
}

/**
 * Read image file and optimize dimensions using HTML5 Canvas (max 1920px)
 */
function readAndOptimizeImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const MAX_DIM = 1920;
        let width = img.width;
        let height = img.height;

        if (width > MAX_DIM || height > MAX_DIM) {
          if (width > height) {
            height = Math.round((height * MAX_DIM) / width);
            width = MAX_DIM;
          } else {
            width = Math.round((width * MAX_DIM) / height);
            height = MAX_DIM;
          }
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);

        const mime = file.type === 'image/png' ? 'image/png' : 'image/jpeg';
        const quality = file.type === 'image/png' ? undefined : 0.88;
        resolve(canvas.toDataURL(mime, quality));
      };
      img.onerror = () => resolve(e.target.result); // Fallback to raw dataUrl
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/**
 * Render attached images in preview bar
 */
function renderAttachedImages() {
  if (!elements.imagePreviewBar) return;

  if (state.attachedImages.length === 0) {
    elements.imagePreviewBar.style.display = 'none';
    elements.imagePreviewBar.innerHTML = '';
    return;
  }

  elements.imagePreviewBar.style.display = 'flex';
  elements.imagePreviewBar.innerHTML = '';

  state.attachedImages.forEach((imgObj) => {
    const item = document.createElement('div');
    item.className = 'preview-item';

    const img = document.createElement('img');
    img.className = 'preview-img';
    img.src = imgObj.dataUrl;
    img.alt = imgObj.name || '미리보기';

    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove-img-btn';
    removeBtn.innerHTML = '✕';
    removeBtn.title = '삭제';
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      removeAttachedImage(imgObj.id);
    });

    item.appendChild(img);
    item.appendChild(removeBtn);
    elements.imagePreviewBar.appendChild(item);
  });
}

/**
 * Remove an attached image by ID
 */
function removeAttachedImage(id) {
  state.attachedImages = state.attachedImages.filter(img => img.id !== id);
  renderAttachedImages();
}

/**
 * Clear all attached images
 */
function clearAttachedImages() {
  state.attachedImages = [];
  if (elements.imagePreviewBar) {
    elements.imagePreviewBar.style.display = 'none';
    elements.imagePreviewBar.innerHTML = '';
  }
  if (elements.imageFileInput) {
    elements.imageFileInput.value = '';
  }
}

/**
 * Open Lightbox modal
 */
function openLightbox(src) {
  if (elements.imageLightboxModal && elements.lightboxImg) {
    elements.lightboxImg.src = src;
    elements.imageLightboxModal.style.display = 'flex';
  }
}

/**
 * Close Lightbox modal
 */
function closeLightbox() {
  if (elements.imageLightboxModal) {
    elements.imageLightboxModal.style.display = 'none';
    if (elements.lightboxImg) elements.lightboxImg.src = '';
  }
}

/**
 * Set Thinking Mode ('fast' | 'deep')
 */
function setThinkingMode(mode) {
  const targetMode = mode === 'deep' ? 'deep' : 'fast';
  state.thinkingMode = targetMode;
  localStorage.setItem('llm_thinking_mode', targetMode);

  // Update topbar buttons
  elements.modeBtns.forEach((btn) => {
    btn.classList.toggle('active', btn.getAttribute('data-mode') === targetMode);
  });

  // Update settings select
  if (elements.thinkingModeSelect) {
    elements.thinkingModeSelect.value = targetMode;
  }
}

/**
 * Load saved preferences from localStorage
 */
function loadSavedPreferences() {
  const savedTheme = localStorage.getItem('llm_theme');
  if (savedTheme === 'light') {
    document.body.classList.remove('dark-theme');
    document.body.classList.add('light-theme');
  }

  const savedMode = localStorage.getItem('llm_thinking_mode') || 'fast';
  setThinkingMode(savedMode);
}

/**
 * Start a clean new chat
 */
function startNewChat() {
  if (state.isGenerating) {
    stopGeneration();
  }
  state.messages = [];
  clearAttachedImages();
  elements.messagesList.innerHTML = '';
  elements.welcomeScreen.style.display = 'block';
  elements.promptInput.value = '';
  elements.promptInput.style.height = 'auto';
  elements.streamStats.style.display = 'none';
  elements.promptInput.focus();
}

/**
 * Handle sending a user prompt (Supporting Multimodal / Vision)
 */
async function handleSendPrompt() {
  let prompt = elements.promptInput.value.trim();
  const hasImages = state.attachedImages.length > 0;

  if (!prompt && !hasImages) return;
  if (state.isGenerating) return;

  // Default prompt for image-only submission
  if (!prompt && hasImages) {
    prompt = '이 이미지를 자세히 분석하고, 무엇이 보이는지 상세하게 묘사해줘.';
  }

  // Hide welcome screen
  elements.welcomeScreen.style.display = 'none';

  // Snapshot attached images
  const attachedImagesSnapshot = [...state.attachedImages];

  // Add User Message to State & UI
  state.messages.push({
    role: 'user',
    content: prompt,
    images: attachedImagesSnapshot,
  });
  appendMessageUI('user', prompt, attachedImagesSnapshot);

  // Clear Input Box & Attached Images
  elements.promptInput.value = '';
  elements.promptInput.style.height = 'auto';
  clearAttachedImages();

  // Prepare Assistant Message Placeholder
  const assistantMsgObj = { role: 'assistant', content: '', reasoning: '' };
  state.messages.push(assistantMsgObj);

  const assistantUI = createAssistantMessageUI();
  elements.messagesList.appendChild(assistantUI.container);
  scrollToBottom();

  // Set Generating State
  setGenerating(true);

  // Stats tracking
  const startTime = performance.now();
  let firstTokenTime = null;
  let totalTokens = 0;
  let hasReasoning = false;

  // Prepare request payload (OpenAI Vision compatible)
  const requestMessages = [];
  if (state.systemPrompt) {
    requestMessages.push({ role: 'system', content: state.systemPrompt });
  }

  // Include recent conversation context
  const recentMessages = state.messages.slice(0, -1).map(m => {
    if (m.images && m.images.length > 0) {
      return {
        role: m.role,
        content: [
          { type: 'text', text: m.content },
          ...m.images.map(img => ({
            type: 'image_url',
            image_url: { url: img.dataUrl }
          }))
        ]
      };
    }
    return {
      role: m.role,
      content: m.content
    };
  });
  requestMessages.push(...recentMessages);

  state.abortController = new AbortController();

  try {
    const isDeepThinking = state.thinkingMode === 'deep';
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: state.activeModel,
        messages: requestMessages,
        temperature: state.temperature,
        max_tokens: state.maxTokens,
        enable_thinking: isDeepThinking,
        stream: true,
      }),
      signal: state.abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    let finishReason = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;

        const dataStr = trimmed.replace(/^data:\s*/, '');
        if (dataStr === '[DONE]') {
          break;
        }

        try {
          const parsed = JSON.parse(dataStr);
          
          if (parsed.error) {
            throw new Error(parsed.error);
          }

          const choice = parsed.choices?.[0];
          if (!choice) continue;

          if (choice.finish_reason) {
            finishReason = choice.finish_reason;
          }

          const delta = choice.delta || {};
          const reasoningChunk = delta.reasoning || delta.reasoning_content;

          // 1. Handle Reasoning Tokens (Thought process from reasoning models)
          if (reasoningChunk) {
            if (!firstTokenTime) firstTokenTime = performance.now();
            hasReasoning = true;
            assistantMsgObj.reasoning += reasoningChunk;
            assistantUI.updateReasoning(assistantMsgObj.reasoning, true);
            totalTokens++;
          }

          // 2. Handle Final Answer Content Tokens
          if (delta.content) {
            if (!firstTokenTime) firstTokenTime = performance.now();
            // If we had reasoning, mark reasoning as finished
            if (hasReasoning && assistantUI.isReasoningActive()) {
              assistantUI.finishReasoning();
            }
            assistantMsgObj.content += delta.content;
            assistantUI.updateContent(assistantMsgObj.content);
            totalTokens++;
          }

          // Update live stream stats
          updateLiveStats(startTime, totalTokens);
          scrollToBottom();

        } catch (jsonErr) {
          if (jsonErr.message && !jsonErr.message.includes('JSON')) {
            throw jsonErr;
          }
        }
      }
    }

    // Stream finished cleanly
    if (hasReasoning) {
      assistantUI.finishReasoning();
    }
    assistantUI.removeCursor();

    // Check if output was cut off due to token limit without producing content
    if (!assistantMsgObj.content.trim()) {
      if (hasReasoning) {
        assistantUI.showWarningCallout(
          '⚠️ 최대 토큰 한도(Max Tokens)에 도달하여 생성이 종료되었습니다.',
          `사고 과정(Reasoning)에서 <strong>${totalTokens}개</strong>의 토큰을 모두 사용하여 본문 답변이 생성되기 전에 종료되었습니다.<br><br>` +
          `💡 <strong>해결 방법:</strong><br>` +
          `• 좌측 하단 ⚙️ <strong>[파라미터 설정]</strong>에서 Max Tokens를 <strong>24576</strong>으로 늘려주세요.<br>` +
          `• 또는 질문 시 <em>"간결하게 핵심 위주로 답변해줘"</em> 등의 시스템 프롬프트를 지정해 보세요.`
        );
      } else {
        assistantUI.showWarningCallout(
          '⚠️ 빈 응답 수신',
          '모델로부터 생성된 내용이 없습니다. 다시 시도해 주세요.'
        );
      }
    }

    const elapsedSec = ((performance.now() - startTime) / 1000).toFixed(2);
    const speedTps = totalTokens > 0 && elapsedSec > 0 ? (totalTokens / elapsedSec).toFixed(1) : 0;
    const finishLabel = finishReason === 'length' ? ' • [토큰한도도달]' : '';
    assistantUI.setMeta(`${totalTokens} 토큰 • ${speedTps} tok/s • ${elapsedSec}초${finishLabel}`);

  } catch (err) {
    if (err.name === 'AbortError') {
      assistantUI.removeCursor();
      assistantUI.setMeta('사용자에 의해 중단됨');
    } else {
      console.error('Chat stream error:', err);
      assistantUI.removeCursor();
      assistantUI.showError(`오류 발생: ${err.message}`);
    }
  } finally {
    setGenerating(false);
    scrollToBottom();
  }
}

/**
 * Stop active generation stream
 */
function stopGeneration() {
  if (state.abortController) {
    state.abortController.abort();
    state.abortController = null;
  }
  setGenerating(false);
}

/**
 * Toggle generating UI state
 */
function setGenerating(isGen) {
  state.isGenerating = isGen;
  elements.sendBtn.classList.toggle('generating', isGen);
  elements.sendBtn.title = isGen ? '생성 중지' : '전송';
  
  const sendIcon = elements.sendBtn.querySelector('.send-icon');
  const stopIcon = elements.sendBtn.querySelector('.stop-icon');
  if (sendIcon && stopIcon) {
    sendIcon.style.display = isGen ? 'none' : 'block';
    stopIcon.style.display = isGen ? 'block' : 'none';
  }

  if (!isGen) {
    elements.tokenCountStat.style.display = 'none';
    elements.speedStat.style.display = 'none';
  }
}

/**
 * Update live streaming stats
 */
function updateLiveStats(startTime, totalTokens) {
  const elapsed = (performance.now() - startTime) / 1000;
  if (elapsed > 0.2 && totalTokens > 0) {
    const tps = (totalTokens / elapsed).toFixed(1);
    elements.tokenCountStat.style.display = 'inline-block';
    elements.speedStat.style.display = 'inline-block';
    elements.tokenCountStat.textContent = `${totalTokens} tok`;
    elements.speedStat.textContent = `${tps} tok/s`;
  }
}

/**
 * Append message element (User with optional attached images or Assistant)
 */
function appendMessageUI(role, content, images = []) {
  const item = document.createElement('div');
  item.className = `message-item ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = role === 'user' ? 'U' : 'AI';

  const body = document.createElement('div');
  body.className = 'message-body';

  // Render attached images gallery if present
  if (images && images.length > 0) {
    const gallery = document.createElement('div');
    gallery.className = 'message-image-gallery';
    images.forEach(imgObj => {
      const imgEl = document.createElement('img');
      imgEl.src = imgObj.dataUrl;
      imgEl.className = 'message-image';
      imgEl.alt = imgObj.name || '첨부 이미지';
      imgEl.title = '클릭하여 확대 보기';
      imgEl.addEventListener('click', () => openLightbox(imgObj.dataUrl));
      gallery.appendChild(imgEl);
    });
    body.appendChild(gallery);
  }

  if (content) {
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = content;
    body.appendChild(bubble);
  }

  item.appendChild(avatar);
  item.appendChild(body);

  elements.messagesList.appendChild(item);
  scrollToBottom();
}

/**
 * Create assistant message element with real-time reasoning & content updates
 */
function createAssistantMessageUI() {
  const item = document.createElement('div');
  item.className = 'message-item assistant';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = '⚡';

  const body = document.createElement('div');
  body.className = 'message-body';

  // Reasoning Container
  let reasoningBox = null;
  let reasoningContent = null;
  let isReasoningActiveFlag = false;

  // Answer Bubble Container
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble markdown-body';
  bubble.innerHTML = '<span class="streaming-cursor"></span>';

  // Meta footer
  const meta = document.createElement('div');
  meta.className = 'message-meta';
  meta.textContent = '';

  body.appendChild(bubble);
  body.appendChild(meta);

  item.appendChild(avatar);
  item.appendChild(body);

  return {
    container: item,
    updateReasoning: (rawReasoningText, isStreaming) => {
      if (!reasoningBox) {
        reasoningBox = document.createElement('div');
        reasoningBox.className = 'reasoning-box';

        const header = document.createElement('div');
        header.className = 'reasoning-header';
        header.innerHTML = `
          <div class="reasoning-title">
            <div class="spinner"></div>
            <span>💭 사고 과정 (Reasoning)</span>
          </div>
          <span class="reasoning-toggle-icon">▼</span>
        `;

        header.addEventListener('click', () => {
          reasoningBox.classList.toggle('collapsed');
        });

        reasoningContent = document.createElement('div');
        reasoningContent.className = 'reasoning-content';

        reasoningBox.appendChild(header);
        reasoningBox.appendChild(reasoningContent);

        // Insert reasoning above the main bubble
        body.insertBefore(reasoningBox, bubble);
        isReasoningActiveFlag = true;
      }

      reasoningContent.textContent = rawReasoningText;
    },
    isReasoningActive: () => isReasoningActiveFlag,
    finishReasoning: () => {
      if (reasoningBox && isReasoningActiveFlag) {
        isReasoningActiveFlag = false;
        const spinner = reasoningBox.querySelector('.spinner');
        if (spinner) spinner.style.display = 'none';
        const titleSpan = reasoningBox.querySelector('.reasoning-title span');
        if (titleSpan) titleSpan.textContent = '💭 사고 과정 완료';
      }
    },
    updateContent: (rawContentText) => {
      bubble.innerHTML = renderMarkdown(rawContentText) + '<span class="streaming-cursor"></span>';
      attachCodeCopyButtons(bubble);
    },
    removeCursor: () => {
      const cursor = bubble.querySelector('.streaming-cursor');
      if (cursor) cursor.remove();
    },
    showError: (errMsg) => {
      bubble.innerHTML = `<div style="color: var(--error); font-weight: 500;">⚠️ ${escapeHtml(errMsg)}</div>`;
    },
    showWarningCallout: (title, descHtml) => {
      bubble.innerHTML = `
        <div class="warning-callout">
          <strong>${escapeHtml(title)}</strong>
          <div class="hint">${descHtml}</div>
        </div>
      `;
    },
    setMeta: (text) => {
      meta.textContent = text;
    }
  };
}

/**
 * Lightweight Markdown Parser for chat responses
 */
function renderMarkdown(md) {
  if (!md) return '';

  let html = md;

  // 1. Code blocks (```language ... ```)
  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const langLabel = lang || 'CODE';
    const escapedCode = escapeHtml(code.trimEnd());
    return `
      <div class="code-block-wrapper">
        <div class="code-block-header">
          <span>${langLabel}</span>
          <button class="copy-code-btn" data-code="${encodeURIComponent(code.trimEnd())}">복사</button>
        </div>
        <pre><code class="language-${langLabel}">${escapedCode}</code></pre>
      </div>
    `;
  });

  // 2. Inline code (`code`)
  html = html.replace(/`([^`]+)`/g, (match, code) => {
    return `<code>${escapeHtml(code)}</code>`;
  });

  // 3. Headers (###, ##, #)
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // 4. Bold and Italic
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // 5. Blockquotes
  html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

  // 6. Unordered lists (- item, * item)
  html = html.replace(/^\s*[-*]\s+(.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // 7. Paragraphs & Linebreaks
  const paragraphs = html.split(/\n\n+/);
  html = paragraphs.map(p => {
    const trimmed = p.trim();
    if (!trimmed) return '';
    if (trimmed.startsWith('<div') || trimmed.startsWith('<h') || trimmed.startsWith('<ul') || trimmed.startsWith('<blockquote')) {
      return trimmed;
    }
    return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
  }).join('');

  return html;
}

/**
 * Attach copy functionality to rendered code blocks
 */
function attachCodeCopyButtons(container) {
  container.querySelectorAll('.copy-code-btn').forEach(btn => {
    if (btn.dataset.attached) return;
    btn.dataset.attached = 'true';
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const code = decodeURIComponent(btn.getAttribute('data-code') || '');
      try {
        await navigator.clipboard.writeText(code);
        btn.textContent = '✓ 복사됨!';
        btn.style.color = 'var(--success)';
        setTimeout(() => {
          btn.textContent = '복사';
          btn.style.color = '';
        }, 2000);
      } catch (err) {
        btn.textContent = '실패';
      }
    });
  });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function scrollToBottom() {
  elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

// Start app on DOM ready
document.addEventListener('DOMContentLoaded', initApp);
