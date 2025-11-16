import './Message.css'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function Message({ content, isUser, type = 'normal', isLoading = false, annotatedImages = [] }) {
  return (
    <div className={`message ${isUser ? 'user' : 'bot'} ${type} ${isLoading ? 'loading' : ''}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div dangerouslySetInnerHTML={{ __html: content }} />
        
        {/* 顯示標註圖片 */}
        {annotatedImages && annotatedImages.length > 0 && (
          <div className="annotated-images-grid">
            {annotatedImages.map((imagePath, idx) => {
              // 從路徑中提取檔案名稱
              const filename = imagePath.split('/').pop()
              const imageUrl = `${API_BASE_URL}/api/annotated-images/${filename}`
              
              return (
                <div key={idx} className="annotated-image-item">
                  <img 
                    src={imageUrl} 
                    alt={`標註圖片 ${idx + 1}`}
                    className="annotated-image"
                  />
                  <div className="image-label">圖片 {idx + 1}</div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

export default Message


