// === CONFIG ===
        const API_URL = "https://aidtm-rag.onrender.com/ask";

        const WELCOME_TEXT = "Hi, I’m AIDTM AI Bot. Ask me anything!";

        // === DOM ===
        const msg = document.getElementById("messages");
        const input = document.getElementById("userInput");

        // === STATE ===
        let chats = JSON.parse(localStorage.getItem("chats") || "{}");
        let currentChat = null;
        let lastUserMessage = "";

        // === HELPERS ===
        function handleKey(e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function createTitle(text) {
            if (!text) return "New Chat";
            let clean = text.trim().replace(/\s+/g, " ");
            if (clean.length > 25) clean = clean.substring(0, 25) + "...";
            return clean;
        }

        function updateTitleIfFirstMessage(msgText) {
            if (!chats[currentChat]) return;
            if (chats[currentChat].title === "New Chat") {
                chats[currentChat].title = createTitle(msgText);
                saveChats();
                loadSidebar();
            }
        }

        function saveChats() {
            localStorage.setItem("chats", JSON.stringify(chats));
        }

        function loadSidebar() {
            let html = "";
            for (let id in chats) {
                html += `
                <div class="chat-item">
                    <div onclick="loadChat(${id})">${chats[id].title}</div>
                    <div class="del-btn" onclick="deleteChat(${id});event.stopPropagation();">✖</div>
                </div>`;
            }
            document.getElementById("chatHistory").innerHTML = html;
        }

        function deleteChat(id) {
            delete chats[id];
            saveChats();
            loadSidebar();
            // if deleting current chat, clear view
            if (String(currentChat) === String(id)) {
                currentChat = null;
                msg.innerHTML = "";
            }
        }

        function newChat() {
            currentChat = Date.now();
            chats[currentChat] = { title: "New Chat", html: "" };
            msg.innerHTML = "";
            // add welcome message immediately (no actions)
            appendWelcomeMessage();
            saveChats();
            loadSidebar();
        }

        function loadChat(id) {
            currentChat = id;
            msg.innerHTML = chats[id].html || "";
            // if no content (maybe user saved empty), ensure welcome exists
            const hasAny = msg.querySelectorAll(".msg").length;
            if (!hasAny) appendWelcomeMessage();
            msg.scrollTop = msg.scrollHeight;
        }

        function showTyping() {
            const div = document.createElement("div");
            div.className = "msg bot";
            div.id = "typing";
            div.innerHTML = `<div class="typing"><span></span><span></span><span></span></div>`;
            msg.appendChild(div);
            msg.scrollTop = msg.scrollHeight;
        }

        function hideTyping() {
            const t = document.getElementById("typing");
            if (t) t.remove();
        }

        function copyText(text) {
            navigator.clipboard.writeText(text);
        }

        // === MESSAGE CREATION ===

        // Append the welcome/default message (no buttons)
        function appendWelcomeMessage() {
            const bubble = document.createElement("div");
            bubble.className = "msg bot welcome";
            bubble.textContent = WELCOME_TEXT;
            msg.appendChild(bubble);
            // save state
            if (currentChat) {
                chats[currentChat].html = msg.innerHTML;
                saveChats();
                loadSidebar();
            }
            msg.scrollTop = msg.scrollHeight;
        }

        // Append a user bubble and attach user actions (copy/edit)
        function appendUserBubble(text) {
            const bubble = document.createElement("div");
            bubble.className = "msg user";
            bubble.textContent = text;

            // append actions container
            const actions = document.createElement("div");
            actions.className = "actions";

            const copyBtn = document.createElement("div");
            copyBtn.className = "action-btn";
            copyBtn.innerHTML = "📋 <span>Copy</span>";
            copyBtn.onclick = () => copyText(text);

            const editBtn = document.createElement("div");
            editBtn.className = "action-btn";
            editBtn.innerHTML = "✏️ <span>Edit</span>";
            editBtn.onclick = () => editMessageInline(bubble, text);

            actions.appendChild(copyBtn);
            actions.appendChild(editBtn);

            bubble.appendChild(actions);
            msg.appendChild(bubble);
            msg.scrollTop = msg.scrollHeight;

            // save
            if (currentChat) {
                chats[currentChat].html = msg.innerHTML;
                saveChats();
            }

            return bubble;
        }

        // Append a bot bubble (regular bot replies) with actions (copy/regenerate)
        function appendBotBubble(text, userMsg) {
            const bubble = document.createElement("div");
            bubble.className = "msg bot";
            bubble.textContent = ""; // will type
            msg.appendChild(bubble);

            // typewriter effect
            let i = 0;
            function type() {
                if (i < text.length) {
                    bubble.textContent = text.substring(0, i + 1);
                    i++;
                    msg.scrollTop = msg.scrollHeight;
                    setTimeout(type, 12);
                } else {
                    // attach actions after typing finishes
                    const actions = document.createElement("div");
                    actions.className = "actions";

                    const copyBtn = document.createElement("div");
                    copyBtn.className = "action-btn";
                    copyBtn.innerHTML = "📋 <span>Copy</span>";
                    copyBtn.onclick = () => copyText(bubble.innerText || bubble.textContent || "");

                    const regenBtn = document.createElement("div");
                    regenBtn.className = "action-btn";
                    regenBtn.innerHTML = "♻️ <span>Regenerate</span>";
                    // regenerate: call regenerate with user message and this bubble
                    regenBtn.onclick = () => regenerate(userMsg, bubble);

                    actions.appendChild(copyBtn);
                    actions.appendChild(regenBtn);
                    bubble.appendChild(actions);

                    // save
                    if (currentChat) {
                        chats[currentChat].html = msg.innerHTML;
                        saveChats();
                    }
                }
            }
            type();
            msg.scrollTop = msg.scrollHeight;
        }

        // === EDIT INLINE with auto-regenerate ===
        async function editMessageInline(bubble, originalText) {
            if (bubble.classList.contains("editing")) return;
            bubble.classList.add("editing");

            bubble.style.background = "#fff4d9";

            // build editor
            const textarea = document.createElement("textarea");
            textarea.className = "inline-editor";
            textarea.value = originalText;
            textarea.style.width = "95%";
            textarea.style.height = "70px";
            textarea.style.padding = "8px";
            textarea.style.fontSize = "14px";
            textarea.style.borderRadius = "6px";
            textarea.style.border = "1px solid #ccc";

            const actionsDiv = document.createElement("div");
            actionsDiv.className = "inline-actions";

            const saveBtn = document.createElement("button");
            saveBtn.className = "save-btn";
            saveBtn.textContent = "Save & Regenerate";

            const cancelBtn = document.createElement("button");
            cancelBtn.className = "cancel-btn";
            cancelBtn.textContent = "Cancel";

            actionsDiv.appendChild(saveBtn);
            actionsDiv.appendChild(cancelBtn);

            // replace bubble content
            bubble.innerHTML = "";
            bubble.appendChild(textarea);
            bubble.appendChild(actionsDiv);
            textarea.focus();

            // Save & regenerate
            saveBtn.onclick = async () => {
                const newText = textarea.value.trim();
                if (!newText) return;

                bubble.classList.remove("editing");
                bubble.innerHTML = ""; // will become user bubble text
                bubble.textContent = newText;

                // re-attach user actions
                const actions = document.createElement("div");
                actions.className = "actions";

                const copyBtn = document.createElement("div");
                copyBtn.className = "action-btn";
                copyBtn.innerHTML = "📋 <span>Copy</span>";
                copyBtn.onclick = () => copyText(newText);

                const editBtn = document.createElement("div");
                editBtn.className = "action-btn";
                editBtn.innerHTML = "✏️ <span>Edit</span>";
                editBtn.onclick = () => editMessageInline(bubble, newText);

                actions.appendChild(copyBtn);
                actions.appendChild(editBtn);
                bubble.appendChild(actions);

                // persist
                if (currentChat) {
                    chats[currentChat].html = msg.innerHTML;
                    saveChats();
                }

                // Remove the immediate next bot reply if it exists (we assume it corresponds)
                const next = bubble.nextElementSibling;
                if (next && next.classList.contains("bot")) {
                    next.remove();
                }

                // send newText to backend and create a new bot reply
                showTyping();
                try {
                    const res = await fetch(API_URL, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query: newText })
                    });
                    const data = await res.json();
                    hideTyping();
                    appendBotBubble(data.answer || "⚠ Server returned empty.", newText);
                } catch (err) {
                    hideTyping();
                    appendBotBubble("⚠ Server error.", newText);
                }
            };

            // Cancel: restore original text
            cancelBtn.onclick = () => {
                bubble.classList.remove("editing");
                bubble.innerHTML = originalText;

                const actions = document.createElement("div");
                actions.className = "actions";

                const copyBtn = document.createElement("div");
                copyBtn.className = "action-btn";
                copyBtn.innerHTML = "📋 <span>Copy</span>";
                copyBtn.onclick = () => copyText(originalText);

                const editBtn = document.createElement("div");
                editBtn.className = "action-btn";
                editBtn.innerHTML = "✏️ <span>Edit</span>";
                editBtn.onclick = () => editMessageInline(bubble, originalText);

                actions.appendChild(copyBtn);
                actions.appendChild(editBtn);
                bubble.appendChild(actions);

                // no change to bot replies or storage
            };
        }

        // === REGENERATE function (used by bot action) ===
        async function regenerate(lastUserMsg, botBubble) {
            // remove existing bot bubble text and show typing indicator in its place
            botBubble.remove();
            showTyping();
            try {
                const res = await fetch(API_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: lastUserMsg })
                });
                const data = await res.json();
                hideTyping();
                appendBotBubble(data.answer || "⚠ Server returned empty.", lastUserMsg);
            } catch (err) {
                hideTyping();
                appendBotBubble("⚠ Server error.", lastUserMsg);
            }
        }

        // === SEND MESSAGE ===
        async function sendMessage() {
            const text = input.value.trim();
            if (!text) return;

            if (!currentChat) {
                // create new chat (this will add welcome then we add real messages)
                newChat();
            }

            // update title if first real user message
            updateTitleIfFirstMessage(text);
            lastUserMessage = text;

            // append user bubble
            const userBubble = appendUserBubble(text);

            // clear input
            input.value = "";
            msg.scrollTop = msg.scrollHeight;

            // call API
            showTyping();
            try {
                const res = await fetch(API_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: text })
                });
                const data = await res.json();
                hideTyping();
                appendBotBubble(data.answer || "⚠ Server returned empty.", text);
            } catch (err) {
                hideTyping();
                appendBotBubble("⚠ Server error.", text);
            }
        }

        // === INITIALIZE APP ===
        function init() {
            loadSidebar();
            // if there's at least one saved chat, load the most recent
            const ids = Object.keys(chats);
            if (ids.length) {
                // load last created chat
                const last = ids[ids.length - 1];
                loadChat(last);
            } else {
                // create a default chat with welcome
                currentChat = Date.now();
                chats[currentChat] = { title: "New Chat", html: "" };
                msg.innerHTML = "";
                appendWelcomeMessage();
                saveChats();
                loadSidebar();
            }
        }

        init();
