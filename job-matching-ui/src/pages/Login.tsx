import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPwd] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null); setLoading(true);
    try {
      await login(email, password);
      nav("/"); // điều hướng tự vào dashboard qua navbar
    } catch (e: any) {
      setErr(e.message || "Lỗi đăng nhập");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <form className="card" onSubmit={onSubmit}>
        <h2>Đăng nhập</h2>
        <label>Email</label>
        <input className="input" value={email} onChange={e=>setEmail(e.target.value)} type="email" required />
        <label className="mt-3">Mật khẩu</label>
        <input className="input" value={password} onChange={e=>setPwd(e.target.value)} type="password" required />
        {err && <p style={{color:'tomato'}} className="mt-3">{err}</p>}
        <button className="btn primary mt-4" disabled={loading}>{loading ? 'Đang đăng nhập…' : 'Đăng nhập'}</button>
        <p className="mt-3">Chưa có tài khoản? <Link to="/register">Đăng ký</Link></p>
      </form>
    </div>
  );
}
