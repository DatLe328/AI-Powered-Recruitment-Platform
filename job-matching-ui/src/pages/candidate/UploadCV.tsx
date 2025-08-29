import { useState } from "react";
import { api } from "../../api/mockApi";
import { useAuth } from "../../context/AuthContext";
import { useNavigate } from "react-router-dom";

function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default function UploadCV() {
  const { user } = useAuth();
  const nav = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("CV mới");
  const [summary, setSummary] = useState("");
  const [skills, setSkills] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  if (!user) return null;

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    if (!file) { setErr("Hãy chọn file"); return; }
    setLoading(true);
    try {
      const b64 = await toBase64(file);
      await api.createCV(user.id, {
        title,
        summary,
        skills: skills.split(',').map(s=>s.trim()).filter(Boolean),
        experience: [],
        fileBase64: b64
      });
      nav("/candidate");
    } catch (e: any) {
      setErr(e.message || "Lỗi upload");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <form className="card" onSubmit={onSubmit}>
        <h2>Upload CV (PDF/DOCX)</h2>
        <label>Tiêu đề CV</label>
        <input className="input" value={title} onChange={e=>setTitle(e.target.value)} required />

        <label className="mt-3">Tóm tắt</label>
        <textarea className="input" value={summary} onChange={e=>setSummary(e.target.value)} rows={4} />

        <label className="mt-3">Kỹ năng (phân tách bằng dấu phẩy)</label>
        <input className="input" value={skills} onChange={e=>setSkills(e.target.value)} placeholder="JavaScript, React, Node.js" />

        <label className="mt-3">Chọn file</label>
        <input className="input" type="file" accept=".pdf,.doc,.docx,.txt" onChange={e=>setFile(e.target.files?.[0] ?? null)} />

        {err && <p style={{color:'tomato'}} className="mt-3">{err}</p>}
        <button className="btn primary mt-4" disabled={loading}>{loading ? 'Đang lưu…' : 'Lưu'}</button>
      </form>
    </div>
  );
}
