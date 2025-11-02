let isApiConnected = false;
let selectedImage = null;

document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    checkApiKeyStatus();
    setInterval(checkApiHealth, 30000);
    
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    
    updateButtonState();
    
    questionInput.addEventListener('input', updateButtonState);
});

function updateButtonState() {
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    
    if (!questionInput || !askButton) return;
    
    const hasText = questionInput.value.trim().length > 0;
    const hasImage = selectedImage !== null;
    
    if (hasText || hasImage) {
        askButton.disabled = false;
    } else {
        askButton.disabled = true;
    }
}

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

function openApiKeyModal() {
    const modal = document.getElementById('apiKeyModal');
    modal.style.display = 'flex';
    checkApiKeyStatus();
}

function closeApiKeyModal() {
    const modal = document.getElementById('apiKeyModal');
    const apiKeyInput = document.getElementById('apiKeyInput');
    modal.style.display = 'none';
    apiKeyInput.value = ''; 
}

async function saveApiKey() {
    const apiKeyInput = document.getElementById('apiKeyInput');
    const apiKey = apiKeyInput.value.trim();
    
    console.log('準備儲存 API Key 到 localStorage...');
    console.log('API Key 長度:', apiKey.length);
    console.log('API Key 開頭:', apiKey.substring(0, 10));
    
    if (!apiKey) {
        alert('請輸入 API Key');
        return;
    }
    
    if (!apiKey.startsWith('sk-')) {
        alert('API Key 格式似乎不正確，應該以 "sk-" 開頭');
        return;
    }
    
    try {
        localStorage.setItem('openai_api_key', apiKey);
        console.log('API Key 已儲存到瀏覽器');
        
        alert('API Key 已成功儲存到瀏覽器！\n\n每次請求時會自動使用此 API Key。');
        
        apiKeyInput.value = '';
        await checkApiKeyStatus();
        closeApiKeyModal();
        
    } catch (error) {
        console.error('儲存 API Key 錯誤:', error);
        alert('儲存失敗：' + error.message);
    }
}

async function clearApiKey() {
    if (!confirm('確定要清除已儲存的 API Key 嗎？\n這會從瀏覽器中刪除 API Key。')) {
        return;
    }
    
    try {
        localStorage.removeItem('openai_api_key');
        console.log('API Key 已從瀏覽器刪除');
        
        alert('API Key 已從瀏覽器中刪除');
        
        document.getElementById('apiKeyInput').value = '';
        checkApiKeyStatus();
        
    } catch (error) {
        console.error('刪除 API Key 錯誤:', error);
        alert(' 刪除失敗：' + error.message);
    }
}

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

async function checkApiKeyStatus() {
    const apiKeyStatus = document.getElementById('apiKeyStatus');
    const statusMessage = document.getElementById('statusMessage');
    const statusIcon = apiKeyStatus.querySelector('.status-icon');
    
    try {
        console.log('檢查瀏覽器中的 API Key 狀態...');
        
        const apiKey = localStorage.getItem('openai_api_key');
        
        if (apiKey && apiKey.startsWith('sk-')) {
            const preview = apiKey.substring(0, 10) + '...';
            apiKeyStatus.classList.add('has-key');
            statusIcon.textContent = '✅';
            statusMessage.textContent = `已設定 API Key (${preview})`;
            console.log('已設定 API Key (儲存在瀏覽器)');
        } else {
            apiKeyStatus.classList.remove('has-key');
            statusIcon.textContent = '🔒';
            statusMessage.textContent = '未設定 API Key';
            console.log('未設定 API Key');
        }
        
    } catch (error) {
        console.error(' 檢查 API Key 狀態錯誤:', error);
        apiKeyStatus.classList.remove('has-key');
        statusIcon.textContent = '❌';
        statusMessage.textContent = '無法檢查狀態';
    }
}

document.addEventListener('click', (event) => {
    const modal = document.getElementById('apiKeyModal');
    if (event.target === modal) {
        closeApiKeyModal();
    }
});

document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        closeApiKeyModal();
    }
});

// ========================================
// ========================================

function addMessage(content, isUser = false, type = 'normal') {
    const welcomeSection = document.getElementById('welcomeSection');
    if (welcomeSection && isUser) {
        welcomeSection.style.display = 'none';
    }
    
    const messagesContainer = document.getElementById('messagesContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'} ${type}`;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = isUser ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function askQuestion() {
    const input = document.getElementById('questionInput');
    const button = document.getElementById('askButton');
    const buttonText = document.getElementById('buttonText');
    const question = input.value.trim();
    
    if (!isApiConnected) {
        alert('❌ 無法連接到後端服務，請檢查後端是否正在運行！');
        return;
    }
    
    if (!question) {
        alert('請輸入問題！');
        return;
    }
    
    const apiKey = localStorage.getItem('openai_api_key');
    if (!apiKey) {
        alert('請先設定 API Key！\n\n點擊右上角的按鈕來設定。');
        openApiKeyModal();
        return;
    }
    
    addMessage(question, true);
    input.value = '';
    
    button.disabled = true;
    buttonText.textContent = '思考中...';
    
    addMessage('<div class="loading-dots"><span></span><span></span><span></span></div>', false);
    
    try {
        console.log('使用瀏覽器中的 API Key 發送請求');
        console.log('DEBUG: apiKey =', apiKey);
        console.log('DEBUG: apiKey 長度 =', apiKey ? apiKey.length : 'null');
        console.log('DEBUG: apiKey 類型 =', typeof apiKey);
        
        const requestBody = {
            question: question,
            top_k: 1,
            api_key: apiKey  
        };
        console.log('DEBUG: 請求體 =', JSON.stringify(requestBody));
        
        const response = await fetch(
            `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.ASK}`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            }
        );
        
        const messagesContainer = document.getElementById('messagesContainer');
        messagesContainer.removeChild(messagesContainer.lastChild);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        addMessage(data.answer);
        
    } catch (error) {
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer.lastChild && messagesContainer.lastChild.querySelector('.loading-dots')) {
            messagesContainer.removeChild(messagesContainer.lastChild);
        }
        
        console.error('錯誤:', error);
        addMessage(
            `❌ 抱歉，發生錯誤：${error.message}<br>` +
            '請稍後再試或檢查 API Key 是否正確。',
            false,
            'error'
        );
    } finally {
        buttonText.textContent = '發送';
        updateButtonState();
    }
}

function askExample(question) {
    document.getElementById('questionInput').value = question;
    updateButtonState();
    askQuestion();
}

function goToHome() {
    console.log('回到主頁功能已觸發');
    
    try {
        // 清除所有訊息
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
            console.log('訊息已清除');
        }
        
        // 顯示歡迎區域
        const welcomeSection = document.getElementById('welcomeSection');
        if (welcomeSection) {
            welcomeSection.style.display = 'block';
            console.log('歡迎區域已顯示');
        }
        
        // 清空輸入框
        const questionInput = document.getElementById('questionInput');
        if (questionInput) {
            questionInput.value = '';
            console.log('輸入框已清空');
        }
        
        // 清除圖片選擇
        if (selectedImage) {
            clearImage();
            console.log('圖片已清除');
        }
        
        // 更新按鈕狀態
        updateButtonState();
        
        // 滾動到頂部
        const chatContainer = document.getElementById('chatContainer');
        if (chatContainer) {
            chatContainer.scrollTop = 0;
            console.log('已滾動到頂部');
        }
        
        console.log('主頁重置完成！');
    } catch (error) {
        console.error('回到主頁時發生錯誤:', error);
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
    }
}

// ========================================
// ========================================

function handleImageSelect(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }
    
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        alert('❌ 不支援的圖片格式！請選擇 JPG、PNG、GIF 或 WEBP 格式的圖片。');
        event.target.value = '';
        return;
    }
    
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSize) {
        alert('❌ 圖片太大！請選擇小於 5MB 的圖片。');
        event.target.value = '';
        return;
    }
    
    selectedImage = file;
    
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
        
        updateButtonState();
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedImage = null;
    document.getElementById('imageInput').value = '';
    document.getElementById('imagePreview').style.display = 'none';
    
    updateButtonState();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function uploadImage() {
    if (!selectedImage) {
        alert('請先選擇圖片！');
        return;
    }
    
    if (!isApiConnected) {
        alert('❌ 無法連接到後端服務，請檢查後端是否正在運行！');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedImage);
    
    addMessage(`📤 正在上傳圖片：${selectedImage.name}...`, true);
    
    addMessage('<div class="loading-dots"><span></span><span></span><span></span></div>', false);
    
    try {
        const response = await fetch(
            `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.UPLOAD}`,
            {
                method: 'POST',
                body: formData
            }
        );
        
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
        
        clearImage();
        
    } catch (error) {
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

const originalAskQuestion = askQuestion;
askQuestion = function() {
    if (selectedImage) {
        uploadImage();
    } else {
        originalAskQuestion();
    }
};
