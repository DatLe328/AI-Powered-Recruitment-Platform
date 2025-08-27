import { useState } from "react";
import { useAuth } from "../../context/AuthContext";
import { api } from "../../api/mockApi";
import type { EmploymentType } from "../../types";
import { useNavigate } from "react-router-dom";

export default function PostJob() {
  const { user } = useAuth();
  const nav = useNavigate();
  const [title, setTitle] = useState("");
  const [description, setDesc] = useState("");
  const [skills, setSkills] = useState("");
  const [salaryMin, setMin] = useState<number | ''>('');
  const [salaryMax, setMax] = useState<number | ''>('');
  const [location, setLoc] = useState("");
  const [employmentType, setType] = useState<EmploymentType>('full-time');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  if (!user) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null); setLoading(true);
    try {
      await api.createJob(user.id, {
        title,
        description,
        skills: skills.split(',').map(s=>s.trim()).filter(Boolean),
        salaryMin: salaryMin === '' ? undefined : Number(salaryMin),
        salaryMax: salaryMax === '' ? undefined : Number(salaryMax),
        location,
        employmentType
      });
      nav("/employer/jobs");
    } catch (e: any) {
      setErr(e.message || "Không thể đăng tin");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <form className="card" onSubmit={submit}>
        <h2>Đăng bài tuyển dụng</h2>
        <label>Chức danh</label>
        <input className="input" value={title} onChange={e=>setTitle(e.target.value)} required />

        <label className="mt-3">Mô tả</label>
        <textarea className="input" rows={6} value={description} onChange={e=>setDesc(e.target.value)} required />

        <label className="mt-3">Kỹ năng (phân tách dấu phẩy)</label>
        <input className="input" value={skills} onChange={e=>setSkills(e.target.value)} placeholder="React, Node.js, SQL" />

        <div className="row mt-3">
          <div>
            <label>Lương tối thiểu</label>
            <input className="input" type="number" value={salaryMin} onChange={e=>setMin(e.target.value === '' ? '' : Number(e.target.value))} />
          </div>
          <div>
            <label>Lương tối đa</label>
            <input className="input" type="number" value={salaryMax} onChange={e=>setMax(e.target.value === '' ? '' : Number(e.target.value))} />
          </div>
        </div>

        <div className="row mt-3">
          <div>
            <label>Địa điểm</label>
            <input className="input" value={location} onChange={e=>setLoc(e.target.value)} />
          </div>
          <div>
            <label>Hình thức</label>
            <select value={employmentType} onChange={e=>setType(e.target.value as EmploymentType)}>
              <option value="full-time">Toàn thời gian</option>
              <option value="part-time">Bán thời gian</option>
              <option value="contract">Hợp đồng</option>
              <option value="intern">Thực tập</option>
            </select>
          </div>
        </div>

        {err && <p style={{color:'tomato'}} className="mt-3">{err}</p>}
        <button className="btn primary mt-4" disabled={loading}>{loading ? 'Đang đăng…' : 'Đăng tuyển'}</button>
      </form>
    </div>
  );
}
