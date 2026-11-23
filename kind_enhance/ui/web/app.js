'use strict';
const $ = id => document.getElementById(id);
let items = [], active = 0, running = false;
const token = document.querySelector('meta[name="local-token"]').content;
const supported = /\.(png|jpe?g|webp|bmp|tiff?)$/i;
function status(message, error = false) { $('status').textContent = message; $('status').classList.toggle('error', error); }
function render() {
  const item = items[active];
  $('before').hidden = !item; $('inputEmpty').hidden = !!item;
  $('after').hidden = !item?.result; $('outputEmpty').hidden = !!item?.result;
  $('download').hidden = !item?.result;
  if (item) {
    $('before').src = item.result?.input || item.preview;
    $('inputName').textContent = item.file.webkitRelativePath || item.file.name;
    if (item.result) {
      $('after').src = item.result.output;
      $('download').href = item.result.output; $('download').download = item.result.filename;
      $('resultInfo').textContent = `${item.result.size[0]} × ${item.result.size[1]} px · ${item.result.seconds.toFixed(2)} giây`;
    } else { $('after').removeAttribute('src'); $('resultInfo').textContent = item.error || 'Chờ xử lý'; }
  } else { $('before').removeAttribute('src'); $('after').removeAttribute('src'); $('inputName').textContent = 'Chưa chọn ảnh'; $('resultInfo').textContent = 'Chờ xử lý'; }
  $('fileList').hidden = items.length < 2; $('listLabel').hidden = items.length < 2;
  $('selectionInfo').textContent = items.length ? `${items.length} ảnh đã chọn` : 'Bạn có thể chọn một ảnh hoặc cả thư mục.';
  $('fileList').value = String(active);
  $('run').disabled = running || !items.some(x => !x.result);
  for (const id of ['chooseFiles', 'chooseFolder', 'fileList']) $(id).disabled = running;
  $('busy').hidden = !running;
}
function choose(files) {
  for (const item of items) URL.revokeObjectURL(item.preview);
  items = Array.from(files).filter(f => supported.test(f.name)).map(file => ({file, preview: URL.createObjectURL(file), result: null, error: ''}));
  active = 0; $('fileList').replaceChildren();
  items.forEach((item, index) => { const option = new Option(item.file.webkitRelativePath || item.file.name, String(index)); $('fileList').add(option); });
  status(items.length ? `Đã chọn ${items.length} ảnh. Nhấn Tăng sáng để xử lý.` : 'Không tìm thấy ảnh phù hợp trong lựa chọn này.', !items.length);
  render();
}
$('chooseFiles').onclick = () => $('files').click();
$('chooseFolder').onclick = () => $('folder').click();
$('files').onchange = event => { choose(event.target.files); event.target.value = ''; };
$('folder').onchange = event => { choose(event.target.files); event.target.value = ''; };
$('fileList').onchange = event => { active = Number(event.target.value); render(); };
$('run').onclick = async () => {
  if (running) return;
  running = true; render();
  for (let index = 0; index < items.length; index++) {
    const item = items[index]; if (item.result) continue;
    active = index; item.error = ''; render();
    status(`Đang xử lý ${index + 1}/${items.length}: ${item.file.name}…`);
    try {
      if (item.file.size > 20 * 1024 * 1024) throw new Error('Ảnh vượt quá 20 MB.');
      const response = await fetch('/api/enhance', {method: 'POST', headers: {'X-Local-Token': token, 'X-File-Name': encodeURIComponent(item.file.name), 'Content-Type': 'application/octet-stream'}, body: item.file});
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Chưa xử lý được ảnh.');
      item.result = data;
      $('fileList').options[index].textContent = '✓ ' + (item.file.webkitRelativePath || item.file.name);
    } catch (error) {
      item.error = error.message || 'Mất kết nối. Kiểm tra cửa sổ chạy local.';
      $('fileList').options[index].textContent = '⚠ ' + item.file.name;
    }
    render();
  }
  running = false; render();
  const completed = items.filter(item => item.result).length;
  status(`Đã xử lý ${completed}/${items.length} ảnh. ${completed < items.length ? 'Có ảnh lỗi; chọn trong danh sách để xem và nhấn Tăng sáng để thử lại.' : 'Chọn ảnh trong danh sách để xem kết quả.'}`, completed < items.length);
};
render();
