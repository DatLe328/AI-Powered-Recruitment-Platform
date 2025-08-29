import { Link } from "react-router-dom";
export default function NotFound() {
  return (
    <div className="container">
      <div className="card">
        <h2>404 - Không tìm thấy</h2>
        <p><Link to="/">Về trang chủ</Link></p>
      </div>
    </div>
  );
}
