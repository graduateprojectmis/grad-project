import './Message.css'

function Message({ content, isUser, type = 'normal' }) {
  return (
    <div className={`message ${isUser ? 'user' : 'bot'} ${type}`}>
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
