import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import type { Role } from "../types";

export default function Register() {
  const { register } = useAuth();
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPwd] = useState("");
  const [fullName, setFullName] = useState("");
  const [role, setRole] = useState<Role>('candidate');
  const [companyName, setCompany] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null); setLoading(true);
    try {
      await register({ email, password, role, fullName: role === 'candidate' ? fullName : undefined, companyName: role === 'employer' ? companyName : undefined });
      nav("/");
    } catch (e: any) {
      setErr(e.message || "Lỗi đăng ký");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <form className="card" onSubmit={onSubmit}>
        <h2>Tạo tài khoản</h2>

        <label>Vai trò</label>
        <select value={role} onChange={e=>setRole(e.target.value as Role)}>
          <option value="candidate">Ứng viên</option>
          <option value="employer">Nhà tuyển dụng</option>
        </select>

        {role === 'candidate' && (
          <>
            <label className="mt-3">Họ tên</label>
            <input className="input" value={fullName} onChange={e=>setFullName(e.target.value)} required />
          </>
        )}

        {role === 'employer' && (
          <>
            <label className="mt-3">Tên công ty</label>
            <input className="input" value={companyName} onChange={e=>setCompany(e.target.value)} required />
          </>
        )}

        <label className="mt-3">Email</label>
        <input className="input" type="email" value={email} onChange={e=>setEmail(e.target.value)} required />

        <label className="mt-3">Mật khẩu</label>
        <input className="input" type="password" value={password} onChange={e=>setPwd(e.target.value)} required />

        {err && <p style={{color:'tomato'}} className="mt-3">{err}</p>}
        <button className="btn primary mt-4" disabled={loading}>{loading ? 'Đang tạo…' : 'Đăng ký'}</button>

        <p className="mt-3">Đã có tài khoản? <Link to="/login">Đăng nhập</Link></p>
      </form>
    </div>
  );
}
