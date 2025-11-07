import './Message.css'

function Message({ content, isUser, type = 'normal', isLoading = false }) {
  return (
    <div className={`message ${isUser ? 'user' : 'bot'} ${type} ${isLoading ? 'loading' : ''}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div 
        className="message-content"
        dangerouslySetInnerHTML={{ __html: content }}
      />
    </div>
  )
}

export default Message
