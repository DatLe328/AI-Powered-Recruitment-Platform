import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../../api/mockApi";
import { useAuth } from "../../context/AuthContext";
import type { CV } from "../../types";

const emptyCV: Omit<CV, 'id' | 'userId' | 'updatedAt'> = {
  title: 'CV JSON',
  summary: '',
  skills: [],
  experience: []
};

export default function EditCV() {
  const { user } = useAuth();
  const nav = useNavigate();
  const { id } = useParams(); // /candidate/cv/:id/edit hoặc /candidate/edit (tạo mới)
  const [loading, setLoading] = useState(false);
  const [cv, setCV] = useState<any>(emptyCV);
  const isNew = useMemo(() => !id, [id]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      if (!user || !id) return;
      setLoading(true);
      try {
        const list = await api.listCVsByUser(user.id);
        const found = list.find(x => x.id === id);
        if (found) {
          const { id: _id, userId: _uid, updatedAt: _u, ...rest } = found;
          setCV(rest);
        } else {
          setErr('Không tìm thấy CV');
        }
      } finally {
        setLoading(false);
      }
    })();
  }, [user, id]);

  const save = async () => {
    setErr(null); setLoading(true);
    try {
      // validate cơ bản
      if (!cv.title || typeof cv.title !== 'string') throw new Error('Thiếu title');
      if (!Array.isArray(cv.skills)) throw new Error('skills phải là mảng');
      if (!Array.isArray(cv.experience)) throw new Error('experience phải là mảng');

      if (isNew) {
        await api.createCV(user!.id, cv);
      } else {
        await api.updateCV(id!, cv);
      }
      nav("/candidate");
    } catch (e: any) {
      setErr(e.message || 'Không thể lưu');
    } finally {
      setLoading(false);
    }
  };

  const pretty = () => setCV(JSON.parse(JSON.stringify(cv, null, 2)));

  return (
    <div className="container">
      <div className="card">
        <h2>{isNew ? 'Tạo CV (JSON)' : 'Chỉnh CV (JSON)'}</h2>
        <p className="mt-2" style={{color:'var(--muted)'}}>
          Bạn có thể chỉnh cấu trúc <code>{`{ title, summary, skills[], experience[] }`}</code>.
        </p>
        <textarea
          className="input mt-3"
          style={{minHeight: 300, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas'}}
          value={typeof cv === 'string' ? cv : JSON.stringify(cv, null, 2)}
          onChange={e => {
            const v = e.target.value;
            try {
              setCV(JSON.parse(v));
              setErr(null);
            } catch {
              setCV(v);
              setErr('JSON chưa hợp lệ (tạm thời)');
            }
          }}
        />
        {err && <p style={{color:'tomato'}} className="mt-3">{err}</p>}
        <div className="mt-3" style={{display:'flex', gap:8, flexWrap:'wrap'}}>
          <button className="btn" onClick={pretty}>Format</button>
          <button className="btn primary" disabled={loading || typeof cv === 'string'} onClick={save}>
            {loading ? 'Đang lưu…' : 'Lưu CV'}
          </button>
        </div>
      </div>
    </div>
  );
}
