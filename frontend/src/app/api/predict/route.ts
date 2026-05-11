import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file');
    
    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }
    
    // Determine the type to route correctly to FastAPI
    const type = file.type.startsWith('video/') ? 'video' : 'image';
    
    // Use the backend URL from environment variables
    const backendUrl = process.env.HF_BACKEND_URL || "https://param20h-deepfake-api.hf.space";
    const targetUrl = `${backendUrl.replace(/\/$/, '')}/predict/${type}`;
    
    const hfToken = process.env.HF_TOKEN;
    
    const headers: Record<string, string> = {};
    if (hfToken) {
      headers['Authorization'] = `Bearer ${hfToken}`;
    }
    
    console.log(`Proxying request to ${targetUrl}...`);
    
    // We proxy the exact FormData over to HF Spaces
    const response = await fetch(targetUrl, {
      method: 'POST',
      headers,
      body: formData,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error("Backend Error:", response.status, errorText);
      return NextResponse.json({ error: `Backend error: ${response.status}` }, { status: response.status });
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Proxy error:", error);
    return NextResponse.json({ error: 'Failed to proxy request to the backend.' }, { status: 500 });
  }
}
