// 全域變數
let isApiConnected = false;
let selectedImage = null;

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    checkApiKeyStatus();
    // 每 30 秒檢查一次連接狀態
    setInterval(checkApiHealth, 30000);
    
    // 監聽輸入框變化，控制發送按鈕狀態
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    
    // 初始化按鈕狀態
    updateButtonState();
    
    // 監聽輸入事件
    questionInput.addEventListener('input', updateButtonState);
});

// 更新發送按鈕狀態
function updateButtonState() {
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    
    if (!questionInput || !askButton) return;
    
    const hasText = questionInput.value.trim().length > 0;
    const hasImage = selectedImage !== null;
    
    // 如果有文字或有圖片，就啟用按鈕
    if (hasText || hasImage) {
        askButton.disabled = false;
    } else {
        askButton.disabled = true;
    }
}

// 檢查 API 健康狀態
async function checkApiHealth() {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`
        );
        
        if (response.ok) {
            const data = await response.json();
            isApiConnected = true;
            statusIndicator.textContent = '🟢';
            statusText.textContent = `已連接 (${data.chroma_db_count} 筆資料)`;
            console.log('API 狀態:', data);
        } else {
            throw new Error('API 回應異常');
        }
    } catch (error) {
        isApiConnected = false;
        statusIndicator.textContent = '🔴';
        statusText.textContent = '連接失敗';
        console.error('無法連接到 API:', error);
    }
}

// ========================================
//  API Key 管理功能
// ========================================

// 打開 API Key 模態框
function openApiKeyModal() {
    const modal = document.getElementById('apiKeyModal');
    modal.style.display = 'flex';
    checkApiKeyStatus();
}

// 關閉 API Key 模態框
function closeApiKeyModal() {
    const modal = document.getElementById('apiKeyModal');
    const apiKeyInput = document.getElementById('apiKeyInput');
    modal.style.display = 'none';
    apiKeyInput.value = ''; // 清空輸入框
}

// 儲存 API Key 到伺服器
async function saveApiKey() {
    const apiKeyInput = document.getElementById('apiKeyInput');
    const apiKey = apiKeyInput.value.trim();
    
    if (!apiKey) {
        alert('❌ 請輸入 API Key');
        return;
    }
    
    if (!apiKey.startsWith('sk-')) {
        alert('⚠️ API Key 格式似乎不正確，應該以 "sk-" 開頭');
        return;
    }
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}/api/config/apikey`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    api_key: apiKey
                })
            }
        );
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '儲存失敗');
        }
        
        const data = await response.json();
        alert('✅ ' + data.message);
        
        // 清空輸入框並更新狀態
        apiKeyInput.value = '';
        checkApiKeyStatus();
        closeApiKeyModal();
        
    } catch (error) {
        console.error('儲存 API Key 錯誤:', error);
        alert('❌ 儲存失敗：' + error.message);
    }
}

// 清除 API Key
async function clearApiKey() {
    if (!confirm('確定要清除已儲存的 API Key 嗎？\n這會從伺服器的 .env 文件中刪除 API Key。')) {
        return;
    }
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}/api/config/apikey`,
            {
                method: 'DELETE'
            }
        );
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '刪除失敗');
        }
        
        const data = await response.json();
        alert('✅ ' + data.message);
        
        // 清空輸入框並更新狀態
        document.getElementById('apiKeyInput').value = '';
        checkApiKeyStatus();
        
    } catch (error) {
        console.error('刪除 API Key 錯誤:', error);
        alert('❌ 刪除失敗：' + error.message);
    }
}

// 切換 API Key 顯示/隱藏
function toggleApiKeyVisibility() {
    const apiKeyInput = document.getElementById('apiKeyInput');
    const visibilityIcon = document.getElementById('visibilityIcon');
    
    if (apiKeyInput.type === 'password') {
        apiKeyInput.type = 'text';
        visibilityIcon.textContent = '👁️‍🗨️';
    } else {
        apiKeyInput.type = 'password';
        visibilityIcon.textContent = '👁️';
    }
}

// 檢查 API Key 狀態
async function checkApiKeyStatus() {
    const apiKeyStatus = document.getElementById('apiKeyStatus');
    const statusMessage = document.getElementById('statusMessage');
    const statusIcon = apiKeyStatus.querySelector('.status-icon');
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}/api/config/apikey/status`
        );
        
        if (!response.ok) {
            throw new Error('無法檢查 API Key 狀態');
        }
        
        const data = await response.json();
        
        if (data.has_key) {
            apiKeyStatus.classList.add('has-key');
            statusIcon.textContent = '✅';
            statusMessage.textContent = `已設定 API Key (${data.key_preview})`;
        } else {
            apiKeyStatus.classList.remove('has-key');
            statusIcon.textContent = '🔒';
            statusMessage.textContent = '未設定 API Key';
        }
        
    } catch (error) {
        console.error('檢查 API Key 狀態錯誤:', error);
        apiKeyStatus.classList.remove('has-key');
        statusIcon.textContent = '❌';
        statusMessage.textContent = '無法檢查狀態';
    }
}

// 點擊模態框背景關閉
document.addEventListener('click', (event) => {
    const modal = document.getElementById('apiKeyModal');
    if (event.target === modal) {
        closeApiKeyModal();
    }
});

// ESC 鍵關閉模態框
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        closeApiKeyModal();
    }
});

// ========================================
//  聊天訊息功能
// ========================================

// 添加訊息到聊天容器
function addMessage(content, isUser = false, type = 'normal') {
    // 隱藏歡迎區域
    const welcomeSection = document.getElementById('welcomeSection');
    if (welcomeSection && isUser) {
        welcomeSection.style.display = 'none';
    }
    
    const messagesContainer = document.getElementById('messagesContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'} ${type}`;
    
    // 創建頭像
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = isUser ? '👤' : '🤖';
    
    // 創建內容
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    // 滾動到底部
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 發送問題
async function askQuestion() {
    const input = document.getElementById('questionInput');
    const button = document.getElementById('askButton');
    const buttonText = document.getElementById('buttonText');
    const question = input.value.trim();
    
    // 檢查連接狀態
    if (!isApiConnected) {
        alert('❌ 無法連接到後端服務，請檢查後端是否正在運行！');
        return;
    }
    
    // 檢查問題是否為空
    if (!question) {
        alert('請輸入問題！');
        return;
    }
    
    // 顯示使用者問題
    addMessage(question, true);
    input.value = '';
    
    // 停用按鈕並顯示載入中
    button.disabled = true;
    buttonText.textContent = '思考中...';
    
    // 清空輸入框後更新按鈕狀態（在處理完成後恢復）
    // 這裡按鈕已經手動設為 disabled，所以不需要立即調用 updateButtonState()
    
    // 顯示載入動畫
    addMessage('<div class="loading-dots"><span></span><span></span><span></span></div>', false);
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.ASK}`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    top_k: 1
                })
            }
        );
        
        // 移除載入動畫
        const messagesContainer = document.getElementById('messagesContainer');
        messagesContainer.removeChild(messagesContainer.lastChild);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        addMessage(data.answer);
        
    } catch (error) {
        // 移除載入動畫（如果還存在）
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer.lastChild && messagesContainer.lastChild.querySelector('.loading-dots')) {
            messagesContainer.removeChild(messagesContainer.lastChild);
        }
        
        console.error('錯誤:', error);
        addMessage(
            `❌ 抱歉，發生錯誤：${error.message}<br>` +
            '請稍後再試或聯繫管理員。',
            false,
            'error'
        );
    } finally {
        buttonText.textContent = '發送';
        // 根據輸入框內容更新按鈕狀態
        updateButtonState();
    }
}

// 使用範例問題
function askExample(question) {
    document.getElementById('questionInput').value = question;
    updateButtonState();
    askQuestion();
}

// 處理 Enter 鍵
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
    }
}

// ========================================
//  圖片上傳功能
// ========================================

// 處理圖片選擇
function handleImageSelect(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }
    
    // 檢查檔案類型
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        alert('❌ 不支援的圖片格式！請選擇 JPG、PNG、GIF 或 WEBP 格式的圖片。');
        event.target.value = '';
        return;
    }
    
    // 檢查檔案大小（限制 5MB）
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSize) {
        alert('❌ 圖片太大！請選擇小於 5MB 的圖片。');
        event.target.value = '';
        return;
    }
    
    selectedImage = file;
    
    // 顯示預覽
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewDiv = document.getElementById('imagePreview');
        const previewImage = document.getElementById('previewImage');
        const imageName = document.getElementById('imageName');
        const imageSize = document.getElementById('imageSize');
        
        previewImage.src = e.target.result;
        imageName.textContent = file.name;
        imageSize.textContent = formatFileSize(file.size);
        previewDiv.style.display = 'block';
        
        // 更新按鈕狀態
        updateButtonState();
    };
    reader.readAsDataURL(file);
}

// 清除選擇的圖片
function clearImage() {
    selectedImage = null;
    document.getElementById('imageInput').value = '';
    document.getElementById('imagePreview').style.display = 'none';
    
    // 更新按鈕狀態
    updateButtonState();
}

// 格式化檔案大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// 上傳圖片
async function uploadImage() {
    if (!selectedImage) {
        alert('請先選擇圖片！');
        return;
    }
    
    // 檢查連接狀態
    if (!isApiConnected) {
        alert('❌ 無法連接到後端服務，請檢查後端是否正在運行！');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedImage);
    
    // 顯示上傳訊息
    addMessage(`📤 正在上傳圖片：${selectedImage.name}...`, true);
    
    // 顯示載入動畫
    addMessage('<div class="loading-dots"><span></span><span></span><span></span></div>', false);
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.UPLOAD}`,
            {
                method: 'POST',
                body: formData
            }
        );
        
        // 移除載入動畫
        const messagesContainer = document.getElementById('messagesContainer');
        messagesContainer.removeChild(messagesContainer.lastChild);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        addMessage(
            `✅ ${data.message}<br>` +
            `檔案名稱：${data.filename}<br>` +
            `檔案大小：${formatFileSize(data.file_size)}<br>` +
            `儲存路徑：${data.file_path}`
        );
        
        // 清除選擇的圖片
        clearImage();
        
    } catch (error) {
        // 移除載入動畫（如果還存在）
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer.lastChild && messagesContainer.lastChild.querySelector('.loading-dots')) {
            messagesContainer.removeChild(messagesContainer.lastChild);
        }
        
        console.error('上傳錯誤:', error);
        addMessage(
            `❌ 上傳失敗：${error.message}<br>` +
            '請檢查網絡連接或稍後再試。',
            false,
            'error'
        );
    }
}

// 修改發送按鈕功能，支援圖片上傳
const originalAskQuestion = askQuestion;
askQuestion = function() {
    // 如果有選擇圖片，則上傳圖片
    if (selectedImage) {
        uploadImage();
    } else {
        originalAskQuestion();
    }
};
