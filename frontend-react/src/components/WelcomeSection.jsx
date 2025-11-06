import { Link2, RotateCcw, Battery, Music } from 'lucide-react'
import './WelcomeSection.css'

function WelcomeSection({ onExampleClick }) {
  const examples = [
    { icon: <Link2 size={20} />, text: '怎麼配對 AirPods？' },
    { icon: <RotateCcw size={20} />, text: '如何重置 AirPods？' },
    { icon: <Battery size={20} />, text: '怎麼確認 AirPods 的電量？' },
    { icon: <Music size={20} />, text: 'AirPods 的降噪功能怎麼用？' },
  ]

  return (
    <div className="welcome-section">
      <div className="welcome-message">
        <div className="welcome-icon">👋</div>
        <h2>您好！我是 AirPods 智慧助手</h2>
        <p className="welcome-description">
          我可以幫您解答關於 AirPods 的各種問題
        </p>
      </div>
      
      <div className="examples">
        <h3>💡 試試這些問題：</h3>
        <div className="example-buttons">
          {examples.map((example, index) => (
            <button
              key={index}
              className="example-button"
              onClick={() => onExampleClick(example.text)}
            >
              <span className="example-icon">{example.icon}</span>
              <span>{example.text}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default WelcomeSection
