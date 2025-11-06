import { useEffect, useRef } from 'react'
import Message from './Message'
import './ChatContainer.css'

function ChatContainer({ messages, children }) {
  const containerRef = useRef(null)

  useEffect(() => {
    // 自動滾動到底部
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [messages])

  return (
    <div className="chat-container" ref={containerRef}>
      <div className="chat-content">
        {children}
        
        <div className="messages-container">
          {messages.map((message) => (
            <Message
              key={message.id}
              content={message.content}
              isUser={message.isUser}
              type={message.type}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

export default ChatContainer
