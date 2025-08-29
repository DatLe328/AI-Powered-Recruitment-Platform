import { Link } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";

export default function EmployerDashboard() {
  const { user } = useAuth();
  if (!user) return null;
  return (
    <div className="container">
      <div className="card">
        <h2>Nhà tuyển dụng • {user.companyName || 'Công ty'}</h2>
        <div className="mt-3" style={{display:'flex', gap:12, flexWrap:'wrap'}}>
          <Link to="/employer/post" className="btn success">＋ Đăng bài tuyển dụng</Link>
          <Link to="/employer/jobs" className="btn">Quản lý tin tuyển dụng</Link>
        </div>
      </div>
    </div>
  );
}
