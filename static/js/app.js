// 页面加载完成后立即执行
document.addEventListener('DOMContentLoaded', () => {
    // 1. 加载已存在的人物列表
    loadPersons();

    // 2. 绑定“添加人物”表单的提交事件
    const addPersonForm = document.getElementById('add-person-form');
    addPersonForm.addEventListener('submit', handleAddPerson);

    // 3. 绑定“在线识别”表单的提交事件
    const recognitionForm = document.getElementById('recognition-form');
    recognitionForm.addEventListener('submit', handleRecognizeFace);
    
    // 4. 为识别图片输入框添加预览功能
    const recognitionImageInput = document.getElementById('recognition-image');
    recognitionImageInput.addEventListener('change', (event) => {
        const preview = document.getElementById('image-preview');
        const file = event.target.files[0];
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
        }
    });
});

// 显示提示信息的辅助函数
function showAlert(message, type = 'danger', containerId) {
    const container = document.getElementById(containerId);
    const wrapper = document.createElement('div');
    wrapper.innerHTML = [
        `<div class="alert alert-${type} alert-dismissible" role="alert">`,
        `   <div>${message}</div>`,
        '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
        '</div>'
    ].join('');
    container.append(wrapper);
}


// 函数：加载人物列表
async function loadPersons() {
    try {
        const response = await fetch('/persons/');
        if (!response.ok) throw new Error('获取人物列表失败');
        
        const persons = await response.json();
        const personListDiv = document.getElementById('person-list');
        personListDiv.innerHTML = ''; // 清空现有列表

        persons.forEach(person => {
            const col = document.createElement('div');
            col.className = 'col';
            col.innerHTML = `
                <div class="card h-100">
                    <img src="${person.photo_path}" class="card-img-top" alt="${person.chinese_name}">
                    <div class="card-body">
                        <h6 class="card-title text-center">${person.chinese_name}</h6>
                    </div>
                </div>
            `;
            personListDiv.appendChild(col);
        });
    } catch (error) {
        console.error('Error loading persons:', error);
        showAlert(error.message, 'danger', 'alert-container-add');
    }
}

// 函数：处理添加人物的表单提交
async function handleAddPerson(event) {
    event.preventDefault(); // 阻止表单默认的刷新页面行为
    
    const form = event.target;
    const formData = new FormData(form);

    try {
        const response = await fetch('/persons/', {
            method: 'POST',
            body: formData,
            // 注意：使用FormData时，浏览器会自动设置正确的Content-Type，无需手动指定
        });

        const result = await response.json();

        if (!response.ok) {
            // 如果后端返回错误信息（如：未检测到人脸）
            throw new Error(result.detail || '添加失败，请检查输入。');
        }
        
        showAlert(`人物 "${result.chinese_name}" 添加成功!`, 'success', 'alert-container-add');
        form.reset(); // 清空表单
        loadPersons(); // 重新加载人物列表
    } catch (error) {
        console.error('Error adding person:', error);
        showAlert(error.message, 'danger', 'alert-container-add');
    }
}

// 函数：处理人脸识别的表单提交
async function handleRecognizeFace(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const resultCard = document.getElementById('recognition-result-card');
    
    try {
        const response = await fetch('/recognize/', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
             throw new Error(result.message || '识别失败');
        }

        // 更新结果显示区域
        document.getElementById('result-name').textContent = result.name;
        document.getElementById('result-similarity').textContent = result.similarity;
        resultCard.style.display = 'block'; // 显示结果卡片

    } catch (error) {
        console.error('Error recognizing face:', error);
        resultCard.style.display = 'none'; // 隐藏结果卡片
        showAlert(error.message, 'danger', 'alert-container-rec');
    }
}