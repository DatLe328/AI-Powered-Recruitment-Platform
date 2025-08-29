import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function NavBar() {
  const { user, logout } = useAuth();

  return (
    <nav style={{borderBottom: '1px solid var(--border)'}}>
      <div className="container" style={{display:'flex', alignItems:'center', gap:16, paddingTop:12, paddingBottom:12}}>
        <Link to="/" className="btn" style={{fontWeight:700}}>Recruit Match</Link>

        <div style={{flex:1}} />

        {!user && (
          <>
            <Link to="/login" className="btn">Đăng nhập</Link>
            <Link to="/register" className="btn primary">Đăng ký</Link>
          </>
        )}

        {user && (
          <>
            {user.role === 'candidate' && <Link to="/candidate" className="btn">Ứng viên</Link>}
            {user.role === 'employer' && <Link to="/employer" className="btn">Nhà tuyển dụng</Link>}
            <span className="badge">Bạn: {user.fullName || user.email} • {user.role}</span>
            <button className="btn" onClick={() => logout()}>Đăng xuất</button>
          </>
        )}
      </div>
    </nav>
  );
}
